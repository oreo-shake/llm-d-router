/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ttftcost

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

func testWeights() Weights {
	// Coefficients where an unmatched long-context tail costs more than a
	// full CPU reload (Method IIB / methodology gate). Units are arbitrary
	// but ranking-stable; offline fit or live BW supplies real values.
	return Weights{
		GPU:                0,
		CPU:                1e-4, // per matched CPU block
		UnmatchedLinear:    1e-5, // w1
		UnmatchedQuadratic: 1e-6, // w2 — dominates at large N
	}
}

func endpointWithMatch(name, producer string, matchBlocks float64, total, blockSize, cached int, tiers map[string]int) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Name: name},
	}, fwkdl.NewMetrics(), nil)
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producer)
	info := attrprefix.NewPrefixCacheMatchInfo(matchBlocks, total, blockSize).
		WithCachedBlockCount(cached)
	if tiers != nil {
		info.WithCachedBlocksByTier(tiers)
	}
	ep.Put(key, info)
	return ep
}

func TestCostFormula(t *testing.T) {
	w := Weights{GPU: 0, CPU: 2, UnmatchedLinear: 3, UnmatchedQuadratic: 4}
	// N=10, m_gpu=6, m_cpu=2 -> m=8
	// cost = 0*6 + 2*2 + 3*(2) + 4*(100-64) = 4 + 6 + 144 = 154
	assert.InDelta(t, 154.0, Cost(w, 10, 6, 2), 1e-9)
	// cold
	assert.InDelta(t, 3*10+4*100, Cost(w, 10, 0, 0), 1e-9)
	// full GPU
	assert.InDelta(t, 0.0, Cost(w, 10, 10, 0), 1e-9)
}

func TestAttributionPrefersGPU(t *testing.T) {
	mGPU, mCPU := Attribution(10, map[string]int{"gpu": 6, "cpu": 10})
	assert.Equal(t, 6, mGPU)
	assert.Equal(t, 4, mCPU)

	mGPU, mCPU = Attribution(10, map[string]int{"cpu": 10})
	assert.Equal(t, 0, mGPU)
	assert.Equal(t, 10, mCPU)
}

func TestScorePrefersFullCPUOverNearFullGPU(t *testing.T) {
	// Gate case from the methodology: ~96k GPU hit vs full ~98k CPU hit.
	// block_size=16 -> N=98304/16=6144 blocks; 96000/16=6000 GPU blocks.
	const (
		blockSize = 16
		nTokens   = 98304
		gpuTokens = 96000
	)
	n := nTokens / blockSize      // 6144
	mGPU := gpuTokens / blockSize // 6000

	producer := "precise-prefix-cache-producer"
	p, err := New(context.Background(), "ttft", Config{
		PrefixMatchInfoProducerName: producer,
		Mode:                        modeOffline,
		Weights:                     testWeights(),
	})
	require.NoError(t, err)

	gpuPod := endpointWithMatch("gpu-pod", producer, float64(mGPU), n, blockSize, mGPU,
		map[string]int{"gpu": mGPU})
	cpuPod := endpointWithMatch("cpu-pod", producer, float64(n), n, blockSize, n,
		map[string]int{"cpu": n})

	scores := p.Score(context.Background(), nil, []fwksched.Endpoint{gpuPod, cpuPod})
	assert.Greater(t, scores[cpuPod], scores[gpuPod],
		"full CPU hit must beat near-full GPU (tail miss); gpu=%v cpu=%v",
		scores[gpuPod], scores[cpuPod])
}

func TestOfflineIgnoresBandwidthUpdate(t *testing.T) {
	p, err := New(context.Background(), "ttft", Config{
		PrefixMatchInfoProducerName: "p",
		Mode:                        modeOffline,
		Weights:                     Weights{CPU: 0.5, UnmatchedLinear: 1, UnmatchedQuadratic: 1},
		Live:                        LiveConfig{BlockBytes: 1024},
	})
	require.NoError(t, err)
	p.UpdateFromOffloadBandwidth(1e9, 1<<30)
	assert.InDelta(t, 0.5, p.CurrentWeights().CPU, 1e-12)
}

func TestLiveUpdatesCPUFromBandwidth(t *testing.T) {
	p, err := New(context.Background(), "ttft", Config{
		PrefixMatchInfoProducerName: "p",
		Mode:                        modeLive,
		Weights:                     Weights{CPU: 0.5, UnmatchedLinear: 1, UnmatchedQuadratic: 1},
		Live: LiveConfig{
			BlockBytes:     1e6,
			MinSampleBytes: 1,
		},
	})
	require.NoError(t, err)
	// 1e6 bytes/block / 1e8 bytes/sec = 0.01 s/block
	p.UpdateFromOffloadBandwidth(1e8, 1e7)
	assert.InDelta(t, 0.01, p.CurrentWeights().CPU, 1e-12)
}

func TestScoreRefreshesCPUFromScrapedCounters(t *testing.T) {
	producer := "precise"
	p, err := New(context.Background(), "ttft", Config{
		PrefixMatchInfoProducerName: producer,
		Mode:                        modeLive,
		Weights:                     Weights{CPU: 0.5, UnmatchedLinear: 1e-5, UnmatchedQuadratic: 1e-6},
		Live: LiveConfig{
			BlockBytes:     1e6,
			MinSampleBytes: 1,
		},
	})
	require.NoError(t, err)

	ep := endpointWithMatch("pod", producer, 10, 10, 16, 10, map[string]int{"cpu": 10})
	ep.Put(attrmetrics.ScalarMetricDataKey(DefaultLoadBytesAttributeKey), attrmetrics.ScalarMetricValue(0))
	ep.Put(attrmetrics.ScalarMetricDataKey(DefaultLoadTimeAttributeKey), attrmetrics.ScalarMetricValue(0))
	_ = p.Score(context.Background(), nil, []fwksched.Endpoint{ep})
	assert.InDelta(t, 0.5, p.CurrentWeights().CPU, 1e-12, "first scrape seeds only")

	ep.Put(attrmetrics.ScalarMetricDataKey(DefaultLoadBytesAttributeKey), attrmetrics.ScalarMetricValue(1e7))
	ep.Put(attrmetrics.ScalarMetricDataKey(DefaultLoadTimeAttributeKey), attrmetrics.ScalarMetricValue(0.1))
	_ = p.Score(context.Background(), nil, []fwksched.Endpoint{ep})
	// 1e6 / (1e7/0.1) = 0.01
	assert.InDelta(t, 0.01, p.CurrentWeights().CPU, 1e-12)
}

func TestFactoryRequiresProducer(t *testing.T) {
	handle := utils.NewTestHandle(context.Background())
	_, err := Factory("x", json.NewDecoder(strings.NewReader(`{"weights":{"cpu":1}}`)), handle)
	require.Error(t, err)
}

func TestFactoryDefaultsToLive(t *testing.T) {
	handle := utils.NewTestHandle(context.Background())
	raw := `{"prefixMatchInfoProducerName":"precise","weights":{"cpu":1,"unmatchedLinear":1,"unmatchedQuadratic":1}}`
	plug, err := Factory("x", json.NewDecoder(strings.NewReader(raw)), handle)
	require.NoError(t, err)
	p := plug.(*Plugin)
	assert.Equal(t, modeLive, p.Mode())
}

func TestNormalizePreservesRanking(t *testing.T) {
	producer := "precise"
	cfg := Config{
		PrefixMatchInfoProducerName: producer,
		Mode:                        modeOffline,
		Weights:                     testWeights(),
		Normalize:                   true,
	}
	p, err := New(context.Background(), "ttft", cfg)
	require.NoError(t, err)

	n := 100
	cold := endpointWithMatch("cold", producer, 0, n, 16, 0, map[string]int{})
	fullGPU := endpointWithMatch("gpu", producer, float64(n), n, 16, n, map[string]int{"gpu": n})
	scores := p.Score(context.Background(), nil, []fwksched.Endpoint{cold, fullGPU})
	assert.Greater(t, scores[fullGPU], scores[cold])
	assert.InDelta(t, 1.0, scores[fullGPU], 1e-9)
	assert.InDelta(t, 0.0, scores[cold], 1e-9)
}

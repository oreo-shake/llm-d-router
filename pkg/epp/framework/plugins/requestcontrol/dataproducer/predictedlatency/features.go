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

package predictedlatency

import (
	"math"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/ttftcost"
)

// Attribute key from core-metrics-extractor customMetrics for
// vllm:kv_offload_active_reload_jobs.
const activeReloadJobsAttrKey = "kv_offload_active_reload_jobs"

// prefixShape is the per-endpoint KV geometry captured at Produce time.
type prefixShape struct {
	score     float64
	n         int
	m         int
	mGPU      int
	mCPU      int
	blockSize int
}

func prefixShapeFromInfo(info *attrprefix.PrefixCacheMatchInfo) prefixShape {
	if info == nil {
		return prefixShape{}
	}
	n := info.TotalBlocks()
	m := info.CachedBlockCount()
	if m < 0 {
		m = 0
	}
	if n > 0 && m > n {
		m = n
	}
	mGPU, mCPU := ttftcost.Attribution(m, info.CachedBlocksByTier())
	bs := info.BlockSizeTokens()
	if bs <= 0 {
		bs = 16
	}
	score := 0.0
	if n > 0 {
		score = info.MatchBlocks() / float64(n)
		if math.IsNaN(score) || score < 0 {
			score = 0
		}
		if score > 1 {
			score = 1
		}
	}
	return prefixShape{
		score:     score,
		n:         n,
		m:         m,
		mGPU:      mGPU,
		mCPU:      mCPU,
		blockSize: bs,
	}
}

func (s prefixShape) unmatchedBlocks() int {
	u := s.n - s.m
	if u < 0 {
		return 0
	}
	return u
}

func buildLatencyShape(
	s prefixShape,
	metrics *fwkdl.Metrics,
	inputTokens int,
	uncached int64,
	ewmaTTFT, ewmaTPOT float64,
	sloTTFT, sloTPOT float64,
	activeReloadJobs float64,
) latencypredictor.LatencyShape {
	bs := s.blockSize
	if bs <= 0 {
		bs = 16
	}
	waiting := 0
	kvFree := 1.0
	if metrics != nil {
		if metrics.WaitingQueueSize > 0 && inputTokens > 0 {
			waiting = metrics.WaitingQueueSize * inputTokens
		}
		kv := metrics.KVCacheUsagePercent
		if kv < 0 {
			kv = 0
		}
		if kv > 1 {
			kv = 1
		}
		kvFree = 1 - kv
	}
	if activeReloadJobs < 0 {
		activeReloadJobs = 0
	}
	unmatched := s.unmatchedBlocks()
	nF := float64(s.n)
	mF := float64(s.m)
	logN := 0.0
	if inputTokens > 0 {
		logN = math.Log1p(float64(inputTokens))
	}
	return latencypredictor.LatencyShape{
		CachedGPUTokens:          s.mGPU * bs,
		CachedCPUTokens:          s.mCPU * bs,
		UnmatchedTokens:          unmatched * bs,
		UnmatchedQuadraticBlocks: nF*nF - mF*mF,
		WaitingTokens:            waiting,
		KVFreeFraction:           kvFree,
		LogInputTokenLength:      logN,
		EwmaTTFTMs:               ewmaTTFT,
		EwmaTPOTMs:               ewmaTPOT,
		SLOTTFTMs:                sloTTFT,
		SLOTPOTMs:                sloTPOT,
		UncachedRequestTokens:    uncached,
		ActiveReloadJobs:         activeReloadJobs,
	}
}

func uncachedTokens(endpoint fwksched.Endpoint, key plugin.DataKey) int64 {
	if endpoint == nil {
		return 0
	}
	raw, ok := endpoint.Get(key)
	if !ok {
		return 0
	}
	u, ok := raw.(*attrconcurrency.UncachedRequestTokens)
	if !ok || u == nil {
		return 0
	}
	return u.Tokens
}

func activeReloadJobs(endpoint fwksched.Endpoint) float64 {
	if endpoint == nil {
		return 0
	}
	raw, ok := endpoint.Get(attrmetrics.ScalarMetricDataKey(activeReloadJobsAttrKey))
	if !ok {
		return 0
	}
	v, ok := raw.(attrmetrics.ScalarMetricValue)
	if !ok {
		return 0
	}
	if float64(v) < 0 {
		return 0
	}
	return float64(v)
}

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

package schedulingdecisionlogger

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/ttftcost"
	"github.com/llm-d/llm-d-router/test/utils"
)

const producer = "precise-prefix-cache-producer"

func testIIBWeights() ttftcost.Weights {
	return ttftcost.Weights{
		GPU:                0,
		CPU:                1e-4,
		UnmatchedLinear:    1e-5,
		UnmatchedQuadratic: 1e-6,
	}
}

func endpointWith(name string, matchBlocks float64, total, cached int, tiers map[string]int) fwksched.Endpoint {
	return endpointWithMetrics(name, matchBlocks, total, cached, tiers, nil)
}

func endpointWithMetrics(name string, matchBlocks float64, total, cached int, tiers map[string]int, metrics *fwkdl.Metrics) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Name: name},
	}, metrics, nil)
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producer)
	info := attrprefix.NewPrefixCacheMatchInfo(matchBlocks, total, 16).WithCachedBlockCount(cached)
	if tiers != nil {
		info.WithCachedBlocksByTier(tiers)
	}
	ep.Put(key, info)
	return ep
}

func schedulingResult(chosen fwksched.Endpoint, scored ...fwksched.Endpoint) *fwksched.SchedulingResult {
	cands := make([]fwksched.ScoredEndpoint, 0, len(scored))
	for i, ep := range scored {
		cands = append(cands, fwksched.ScoredEndpoint{Endpoint: ep, Score: float64(len(scored) - i)})
	}
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {
				TargetEndpoints:  []fwksched.Endpoint{chosen},
				ScoredCandidates: cands,
			},
		},
	}
}

func newTestPlugin(t *testing.T, cfg Config) *Plugin {
	t.Helper()
	if cfg.Experiment == "" {
		cfg.Experiment = "test-exp"
	}
	if cfg.PrefixMatchInfoProducerName == "" {
		cfg.PrefixMatchInfoProducerName = producer
	}
	reg := prometheus.NewRegistry()
	h := plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(reg))
	p, err := New("logger", cfg, h)
	require.NoError(t, err)
	return p
}

func TestFactoryRequiresFields(t *testing.T) {
	_, err := Factory("x", json.NewDecoder(strings.NewReader(`{}`)), nil)
	require.Error(t, err)

	_, err = Factory("x", json.NewDecoder(strings.NewReader(
		`{"experiment":"e"}`)), nil)
	require.Error(t, err)

	p, err := Factory("x", json.NewDecoder(strings.NewReader(
		`{"experiment":"e","prefixMatchInfoProducerName":"precise"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, p.TypedName().Type)
}

func TestAgreeMxWWhenChosenIsMaxMatchBlocks(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTestPlugin(t, Config{
		IIBWeights: ptrWeights(testIIBWeights()),
		LogAgrees:  true,
	})

	gpu := endpointWith("gpu", 100, 110, 100, map[string]int{"gpu": 100})
	cpu := endpointWith("cpu", 50, 110, 50, map[string]int{"cpu": 50})
	res := schedulingResult(gpu, gpu, cpu)

	rec, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "r1"}, res)
	require.NoError(t, err)
	require.NotNil(t, rec.AgreeMxW)
	assert.True(t, *rec.AgreeMxW)
	assert.Equal(t, gpu.GetMetadata().ID.String(), rec.Counterfactual.MxW.Endpoint)
	assert.Equal(t, gpu.GetMetadata().ID.String(), rec.Chosen)
	assert.False(t, rec.GateCase)

	require.NoError(t, p.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "r1"}, res))
}

func TestDisagreeGateCaseNearFullGPUVsFullCPU(t *testing.T) {
	// Methodology gate under tier-weighted MatchBlocks: near-full GPU scores
	// higher for m×w than a full CPU hit (CPU tier attenuated), while IIB cost
	// prefers the full CPU reload over the unmatched GPU tail.
	const n, mGPU = 6144, 6000
	gpu := endpointWith("gpu-pod", float64(mGPU), n, mGPU, map[string]int{"gpu": mGPU})
	// CPU tier weight ~0.5 -> MatchBlocks < GPU's 6000 even though m=N.
	cpu := endpointWith("cpu-pod", float64(n)*0.5, n, n, map[string]int{"cpu": n})

	p := newTestPlugin(t, Config{IIBWeights: ptrWeights(testIIBWeights())})
	// Live choice follows m×w (gpu).
	res := schedulingResult(gpu, gpu, cpu)

	rec, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "gate"}, res)
	require.NoError(t, err)
	require.NotNil(t, rec.Counterfactual.MxW)
	require.NotNil(t, rec.Counterfactual.IIB)
	assert.Equal(t, gpu.GetMetadata().ID.String(), rec.Counterfactual.MxW.Endpoint)
	assert.Equal(t, cpu.GetMetadata().ID.String(), rec.Counterfactual.IIB.Endpoint)
	require.NotNil(t, rec.AgreeMxW)
	assert.True(t, *rec.AgreeMxW)
	require.NotNil(t, rec.AgreeIIB)
	assert.False(t, *rec.AgreeIIB)
	assert.True(t, rec.GateCase)
	assert.True(t, disagreesWithAny(rec.Chosen, rec.Counterfactual))
}

func TestMissingPrefixInfoSkipped(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newTestPlugin(t, Config{})

	bare := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID: k8stypes.NamespacedName{Name: "bare"},
	}, nil, nil)
	res := schedulingResult(bare, bare)
	require.NoError(t, p.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "miss"}, res))

	_, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "miss"}, res)
	require.Error(t, err)
}

func TestMissingScoredCandidatesFallsBackToTargets(t *testing.T) {
	p := newTestPlugin(t, Config{IIBWeights: ptrWeights(testIIBWeights()), LogAgrees: true})
	gpu := endpointWith("gpu", 10, 10, 10, map[string]int{"gpu": 10})
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{gpu}},
		},
	}
	rec, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "fb"}, res)
	require.NoError(t, err)
	assert.Len(t, rec.Candidates, 1)
	assert.Equal(t, gpu.GetMetadata().ID.String(), rec.Chosen)
	assert.Empty(t, rec.ScorerScores)
}

func TestJSONLWriterAndSkipAgrees(t *testing.T) {
	ctx := utils.NewTestContext(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.jsonl")

	w := testIIBWeights()
	p := newTestPlugin(t, Config{
		OutputPath: path,
		IIBWeights: &w,
		LogAgrees:  false,
	})

	const n, mGPU = 6144, 6000
	gpu := endpointWith("gpu-pod", float64(mGPU), n, mGPU, map[string]int{"gpu": mGPU})
	cpu := endpointWith("cpu-pod", float64(n)*0.5, n, n, map[string]int{"cpu": n})

	// Agree with both (single candidate): should not write when logAgrees=false.
	only := endpointWith("only", 10, 10, 10, map[string]int{"gpu": 10})
	require.NoError(t, p.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "agree"}, schedulingResult(only, only)))

	// Disagree with IIB: should write.
	require.NoError(t, p.PreRequest(ctx, &fwksched.InferenceRequest{RequestID: "dis"}, schedulingResult(gpu, gpu, cpu)))

	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	require.NoError(t, sc.Err())
	require.Len(t, lines, 1)

	var rec DecisionRecord
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &rec))
	assert.Equal(t, "dis", rec.RequestID)
	assert.True(t, rec.GateCase)
}

func TestPickMxWTieBreak(t *testing.T) {
	cands := []CandidateFeatures{
		{Endpoint: "b", MatchBlocks: 5},
		{Endpoint: "a", MatchBlocks: 5},
	}
	pick := pickMxW(cands)
	require.NotNil(t, pick)
	assert.Equal(t, "a", pick.Endpoint)
}

func TestCounterfactualProfilesConfigValidation(t *testing.T) {
	_, err := New("logger", Config{
		Experiment:                  "e",
		PrefixMatchInfoProducerName: producer,
		CounterfactualProfiles: []CounterfactualProfileConfig{
			{Name: "ttft-only", Scorers: []ScorerRef{{Type: componentTTFT}}},
		},
	}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "iibWeights")

	_, err = New("logger", Config{
		Experiment:                  "e",
		PrefixMatchInfoProducerName: producer,
		IIBWeights:                  ptrWeights(testIIBWeights()),
		CounterfactualProfiles: []CounterfactualProfileConfig{
			{Name: "bad", Scorers: []ScorerRef{{Type: "nope"}}},
		},
	}, nil)
	require.Error(t, err)

	_, err = New("logger", Config{
		Experiment:                  "e",
		PrefixMatchInfoProducerName: producer,
		CounterfactualProfiles: []CounterfactualProfileConfig{
			{Name: "a", Scorers: []ScorerRef{{Type: componentPrefix}}},
			{Name: "a", Scorers: []ScorerRef{{Type: componentQueue}}},
		},
	}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

func TestCounterfactualProfilesPrefixVsTTFTWithQueueKV(t *testing.T) {
	// Same geometry as the gate case: m×w/prefix prefers GPU; IIB/ttft prefers CPU.
	// Equal queue+kv so the combo disagreement is driven by the cache component.
	const n, mGPU = 6144, 6000
	gpu := endpointWithMetrics("gpu-pod", float64(mGPU), n, mGPU, map[string]int{"gpu": mGPU},
		&fwkdl.Metrics{WaitingQueueSize: 1, KVCacheUsagePercent: 0.4})
	cpu := endpointWithMetrics("cpu-pod", float64(n)*0.5, n, n, map[string]int{"cpu": n},
		&fwkdl.Metrics{WaitingQueueSize: 1, KVCacheUsagePercent: 0.4})

	p := newTestPlugin(t, Config{
		IIBWeights: ptrWeights(testIIBWeights()),
		CounterfactualProfiles: []CounterfactualProfileConfig{
			{
				Name: "prefix-q-kv",
				Scorers: []ScorerRef{
					{Type: componentPrefix},
					{Type: componentQueue},
					{Type: componentKVUtil},
				},
			},
			{
				Name: "ttft-q-kv",
				Scorers: []ScorerRef{
					{Type: componentTTFT},
					{Type: componentQueue},
					{Type: componentKVUtil},
				},
			},
		},
	})
	res := schedulingResult(gpu, gpu, cpu)
	rec, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "cf"}, res)
	require.NoError(t, err)

	require.Contains(t, rec.Counterfactual.Profiles, "prefix-q-kv")
	require.Contains(t, rec.Counterfactual.Profiles, "ttft-q-kv")
	assert.Equal(t, gpu.GetMetadata().ID.String(), rec.Counterfactual.Profiles["prefix-q-kv"].Endpoint)
	assert.Equal(t, cpu.GetMetadata().ID.String(), rec.Counterfactual.Profiles["ttft-q-kv"].Endpoint)
	assert.True(t, rec.ProfilesDisagree)
	require.NotNil(t, rec.AgreeProfiles["prefix-q-kv"])
	assert.True(t, *rec.AgreeProfiles["prefix-q-kv"])
	require.NotNil(t, rec.AgreeProfiles["ttft-q-kv"])
	assert.False(t, *rec.AgreeProfiles["ttft-q-kv"])
	assert.True(t, disagreesWithAny(rec.Chosen, rec.Counterfactual))

	require.Equal(t, 1, rec.Candidates[0].WaitingQueueSize)
	require.InDelta(t, 0.4, rec.Candidates[0].KVCacheUsagePercent, 1e-9)
}

func TestQueueComponentCanFlipPrefixRanking(t *testing.T) {
	// Prefix alone prefers a; heavy queue on a flips prefix+queue to b.
	a := endpointWithMetrics("a", 10, 10, 10, map[string]int{"gpu": 10},
		&fwkdl.Metrics{WaitingQueueSize: 20, KVCacheUsagePercent: 0.5})
	b := endpointWithMetrics("b", 8, 10, 8, map[string]int{"gpu": 8},
		&fwkdl.Metrics{WaitingQueueSize: 0, KVCacheUsagePercent: 0.5})

	p := newTestPlugin(t, Config{
		CounterfactualProfiles: []CounterfactualProfileConfig{
			{Name: "prefix-only", Scorers: []ScorerRef{{Type: componentPrefix}}},
			{Name: "prefix-q", Scorers: []ScorerRef{
				{Type: componentPrefix, Weight: 1},
				{Type: componentQueue, Weight: 1},
			}},
		},
	})
	res := schedulingResult(a, a, b)
	rec, err := p.buildRecord(&fwksched.InferenceRequest{RequestID: "q"}, res)
	require.NoError(t, err)
	assert.Equal(t, a.GetMetadata().ID.String(), rec.Counterfactual.Profiles["prefix-only"].Endpoint)
	assert.Equal(t, b.GetMetadata().ID.String(), rec.Counterfactual.Profiles["prefix-q"].Endpoint)
	assert.True(t, rec.ProfilesDisagree)
}

func ptrWeights(w ttftcost.Weights) *ttftcost.Weights { return &w }

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

// Package ttftcost scores endpoints by predicted TTFT cost (Method IIB).
//
//	cost  = w_gpu*m_gpu + w_cpu*m_cpu + w1*(N-m) + w2*(N^2-m^2)
//	score = -cost
//
// N, m, m_gpu, and m_cpu are in KV blocks. w_gpu is typically 0.
package ttftcost

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	// PluginType is the unique identifier for the TTFT cost scorer.
	PluginType = "ttft-cost-scorer"

	modeLive    = "live"
	modeOffline = "offline"

	tierGPU = "gpu"
	tierCPU = "cpu"
)

// Weights holds Method IIB cost coefficients (block units).
type Weights struct {
	// GPU is w_gpu: cost per matched GPU block. Default 0.
	GPU float64 `json:"gpu"`
	// CPU is w_cpu: cost per matched CPU block (reload).
	CPU float64 `json:"cpu"`
	// UnmatchedLinear is w1: linear cost of unmatched blocks (N-m).
	UnmatchedLinear float64 `json:"unmatchedLinear"`
	// UnmatchedQuadratic is w2: quadratic cost of unmatched attention (N^2-m^2).
	UnmatchedQuadratic float64 `json:"unmatchedQuadratic"`
}

// LiveConfig controls how live mode refreshes Weights from engine metrics.
type LiveConfig struct {
	// BlockBytes is KV bytes transferred per block when deriving w_cpu from
	// average offload bandwidth. Required for live w_cpu updates.
	BlockBytes float64 `json:"blockBytes,omitempty"`
	// MinSampleBytes ignores bandwidth samples below this many transferred
	// bytes in the scrape window (noise floor). Default 1<<20 (1 MiB).
	MinSampleBytes float64 `json:"minSampleBytes,omitempty"`
	// LoadBytesAttributeKey is the core-metrics-extractor attribute for
	// vllm:kv_offload_load_bytes_total. Default kv_offload_load_bytes.
	LoadBytesAttributeKey string `json:"loadBytesAttributeKey,omitempty"`
	// LoadTimeAttributeKey is the core-metrics-extractor attribute for
	// vllm:kv_offload_load_time_total. Default kv_offload_load_time.
	LoadTimeAttributeKey string `json:"loadTimeAttributeKey,omitempty"`
}

// Config is the JSON parameters for ttft-cost-scorer.
type Config struct {
	// PrefixMatchInfoProducerName selects which PrefixCacheMatchInfo producer
	// to read. Required.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
	// Mode is "live" (default) or "offline". Offline freezes Weights from
	// config. Live starts from Weights and may refresh CPU coefficients from
	// scraped offload metrics.
	Mode string `json:"mode,omitempty"`
	// Weights are the offline coefficients and the live-mode starting point.
	Weights Weights `json:"weights"`
	// Live configures live coefficient updates. Ignored when Mode is offline.
	Live LiveConfig `json:"live,omitempty"`
	// Normalize maps -cost into approximately [0,1] via
	// 1 - cost/cost_cold(N). Ranking is unchanged. Default false so the
	// score matches the Method IIB cost form for prefix-only A/B.
	Normalize bool `json:"normalize,omitempty"`
}

// Plugin implements fwksched.Scorer for Method IIB TTFT cost.
type Plugin struct {
	typedName          plugin.TypedName
	prefixMatchDataKey plugin.DataKey
	normalize          bool
	mode               string
	live               LiveConfig
	loadBytesKey       string
	loadTimeKey        string
	weights            atomic.Pointer[Weights]

	sampleMu  sync.Mutex
	hasSample bool
	lastBytes float64
	lastSecs  float64
}

var _ fwksched.Scorer = &Plugin{}

// Factory constructs a ttft-cost-scorer from JSON parameters.
func Factory(name string, decoder *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{
		Mode: modeLive,
		Weights: Weights{
			GPU: 0,
		},
		Live: LiveConfig{
			MinSampleBytes: 1 << 20,
		},
	}
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to decode %s parameters: %w", PluginType, err)
		}
	}
	return New(handle.Context(), name, cfg)
}

// New builds a Plugin. Used by Factory and tests.
func New(_ context.Context, name string, cfg Config) (*Plugin, error) {
	if cfg.PrefixMatchInfoProducerName == "" {
		return nil, fmt.Errorf("%s: prefixMatchInfoProducerName is required", PluginType)
	}
	mode := cfg.Mode
	if mode == "" {
		mode = modeLive
	}
	if mode != modeLive && mode != modeOffline {
		return nil, fmt.Errorf("%s: mode must be %q or %q, got %q", PluginType, modeLive, modeOffline, mode)
	}
	if cfg.Weights.CPU < 0 || cfg.Weights.UnmatchedLinear < 0 ||
		cfg.Weights.UnmatchedQuadratic < 0 || cfg.Weights.GPU < 0 {
		return nil, fmt.Errorf("%s: weights must be >= 0", PluginType)
	}
	live := cfg.Live
	if live.MinSampleBytes <= 0 {
		live.MinSampleBytes = 1 << 20
	}
	loadBytesKey := live.LoadBytesAttributeKey
	if loadBytesKey == "" {
		loadBytesKey = DefaultLoadBytesAttributeKey
	}
	loadTimeKey := live.LoadTimeAttributeKey
	if loadTimeKey == "" {
		loadTimeKey = DefaultLoadTimeAttributeKey
	}

	p := &Plugin{
		typedName: plugin.TypedName{
			Type: PluginType,
			Name: name,
		},
		prefixMatchDataKey: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(
			cfg.PrefixMatchInfoProducerName,
		),
		normalize:    cfg.Normalize,
		mode:         mode,
		live:         live,
		loadBytesKey: loadBytesKey,
		loadTimeKey:  loadTimeKey,
	}
	w := cfg.Weights
	p.weights.Store(&w)
	return p, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

// Category returns Affinity: prefer endpoints with lower predicted TTFT cost.
func (p *Plugin) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Produces returns no data keys.
func (p *Plugin) Produces() map[plugin.DataKey]any { return map[plugin.DataKey]any{} }

// Consumes requires PrefixCacheMatchInfo from the configured producer.
func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{p.prefixMatchDataKey: attrprefix.PrefixCacheMatchInfo{}},
	}
}

// CurrentWeights returns a copy of the active coefficients.
func (p *Plugin) CurrentWeights() Weights {
	w := p.weights.Load()
	if w == nil {
		return Weights{}
	}
	return *w
}

// SetWeights replaces the active coefficients. Safe for concurrent use.
func (p *Plugin) SetWeights(w Weights) {
	cp := w
	p.weights.Store(&cp)
}

// Mode returns "live" or "offline".
func (p *Plugin) Mode() string { return p.mode }

// UpdateFromOffloadBandwidth refreshes w_cpu from average load bandwidth
// (bytes/sec) when Mode is live. No-op in offline mode or when BlockBytes
// is unset / sample too small.
func (p *Plugin) UpdateFromOffloadBandwidth(bytesPerSec float64, sampleBytes float64) {
	if p.mode != modeLive {
		return
	}
	if p.live.BlockBytes <= 0 || bytesPerSec <= 0 || sampleBytes < p.live.MinSampleBytes {
		return
	}
	// Seconds to reload one block at the observed average bandwidth.
	wCPU := p.live.BlockBytes / bytesPerSec
	cur := p.CurrentWeights()
	cur.CPU = wCPU
	p.SetWeights(cur)
}

// Cost returns the Method IIB cost for one endpoint's match breakdown.
func Cost(w Weights, n, mGPU, mCPU int) float64 {
	if n < 0 {
		n = 0
	}
	if mGPU < 0 {
		mGPU = 0
	}
	if mCPU < 0 {
		mCPU = 0
	}
	m := mGPU + mCPU
	if m > n {
		m = n
	}
	unmatched := float64(n - m)
	nF := float64(n)
	mF := float64(m)
	return w.GPU*float64(mGPU) +
		w.CPU*float64(mCPU) +
		w.UnmatchedLinear*unmatched +
		w.UnmatchedQuadratic*(nF*nF-mF*mF)
}

// Attribution turns producer tier maps into an exclusive (m_gpu, m_cpu)
// partition of the unweighted consecutive match length m.
// GPU blocks are preferred (w_gpu is typically 0).
func Attribution(m int, tiers map[string]int) (mGPU, mCPU int) {
	if m < 0 {
		m = 0
	}
	gpu := 0
	if tiers != nil {
		gpu = tiers[tierGPU]
	}
	if gpu < 0 {
		gpu = 0
	}
	mGPU = gpu
	if mGPU > m {
		mGPU = m
	}
	mCPU = m - mGPU
	return mGPU, mCPU
}

// Score returns -cost (or a normalized benefit) per endpoint.
func (p *Plugin) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	p.refreshLiveFromEndpoints(endpoints)

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)
	w := p.CurrentWeights()
	missing := 0

	for _, endpoint := range endpoints {
		scores[endpoint] = 0.0
		raw, ok := endpoint.Get(p.prefixMatchDataKey)
		if !ok {
			missing++
			continue
		}
		info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
		if !ok {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type, assigning score 0",
				"endpoint", endpoint)
			continue
		}
		n := info.TotalBlocks()
		if n <= 0 {
			continue
		}
		m := info.CachedBlockCount()
		if m < 0 {
			m = 0
		}
		if m > n {
			m = n
		}
		mGPU, mCPU := Attribution(m, info.CachedBlocksByTier())
		cost := Cost(w, n, mGPU, mCPU)
		if p.normalize {
			cold := Cost(w, n, 0, 0)
			if cold > 0 {
				scores[endpoint] = 1.0 - cost/cold
			} else {
				scores[endpoint] = -cost
			}
			continue
		}
		scores[endpoint] = -cost
	}

	if missing > 0 {
		logger.V(logutil.DEFAULT).Info("PrefixCacheMatchInfo not found for endpoints, assigning score 0",
			"count", missing, "key", p.prefixMatchDataKey.String())
	}
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("ttft-cost scores", "weights", w, "mode", p.mode, "normalize", p.normalize)
	}
	return scores
}

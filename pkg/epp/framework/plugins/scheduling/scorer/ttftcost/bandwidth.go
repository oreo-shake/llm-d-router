/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    10|Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ttftcost

import (
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// Default attribute keys published by core-metrics-extractor customMetrics
// for vLLM SimpleCPUOffload / OffloadingConnector load counters.
const (
	DefaultLoadBytesAttributeKey = "kv_offload_load_bytes"
	DefaultLoadTimeAttributeKey  = "kv_offload_load_time"
)

// BandwidthSample holds a scrape-window delta of offload transfer counters.
type BandwidthSample struct {
	// Bytes is delta of vllm:kv_offload_load_bytes over the window.
	Bytes float64
	// Seconds is delta of vllm:kv_offload_load_time over the window.
	Seconds float64
}

// BytesPerSecond returns average load bandwidth, or 0 if the sample is empty.
func (s BandwidthSample) BytesPerSecond() float64 {
	if s.Seconds <= 0 || s.Bytes <= 0 {
		return 0
	}
	return s.Bytes / s.Seconds
}

// ApplyBandwidthSample updates live w_cpu from a scrape-window sample.
// No-op in offline mode.
func (p *Plugin) ApplyBandwidthSample(sample BandwidthSample) {
	p.UpdateFromOffloadBandwidth(sample.BytesPerSecond(), sample.Bytes)
}

// refreshLiveFromEndpoints reads scraped load counters from endpoint attributes,
// aggregates across candidates, and refreshes w_cpu from the scrape-window delta.
// First observation only seeds the baseline. Counter resets reseed without applying.
func (p *Plugin) refreshLiveFromEndpoints(endpoints []fwksched.Endpoint) {
	if p.mode != modeLive || p.live.BlockBytes <= 0 {
		return
	}

	var sumBytes, sumSecs float64
	found := 0
	for _, endpoint := range endpoints {
		b, okB := readScalarAttr(endpoint, p.loadBytesKey)
		s, okS := readScalarAttr(endpoint, p.loadTimeKey)
		if !okB || !okS {
			continue
		}
		sumBytes += b
		sumSecs += s
		found++
	}
	if found == 0 {
		return
	}

	p.sampleMu.Lock()
	defer p.sampleMu.Unlock()

	if !p.hasSample {
		p.lastBytes = sumBytes
		p.lastSecs = sumSecs
		p.hasSample = true
		return
	}

	dB := sumBytes - p.lastBytes
	dS := sumSecs - p.lastSecs
	p.lastBytes = sumBytes
	p.lastSecs = sumSecs
	if dB < 0 || dS < 0 {
		// Pod restart / counter reset: reseed only.
		return
	}
	p.ApplyBandwidthSample(BandwidthSample{Bytes: dB, Seconds: dS})
}

func readScalarAttr(endpoint fwksched.Endpoint, key string) (float64, bool) {
	if key == "" {
		return 0, false
	}
	raw, ok := endpoint.Get(attrmetrics.ScalarMetricDataKey(key))
	if !ok {
		return 0, false
	}
	v, ok := raw.(attrmetrics.ScalarMetricValue)
	if !ok {
		return 0, false
	}
	return float64(v), true
}

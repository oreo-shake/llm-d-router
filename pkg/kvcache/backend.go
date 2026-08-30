/*
Copyright 2025 The llm-d Authors.

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

package kvcache

import (
	"fmt"
	"sort"
)

// PrefixTokenWeight maps a consecutive prefix length in tokens to a scoring
// weight for one device tier. The scorer looks up a single weight from the
// matched token count on that tier, then applies it to every block of the
// tier in the consecutive prefix.
type PrefixTokenWeight struct {
	// MaxTokens is the inclusive upper bound of this bucket, in tokens.
	// Zero is a catch-all for lengths above every positive bound.
	MaxTokens int `json:"maxTokens"`
	// Weight is the scoring weight used when the matched prefix on this
	// tier falls in the bucket.
	Weight float64 `json:"weight"`
}

// KVCacheBackendConfig is the scoring weight for one KV-cache device tier.
type KVCacheBackendConfig struct {
	// Name is the identifier for this medium (e.g., "gpu", "cpu", "disk")
	Name string `json:"name"`
	// Weight is the scoring weight for blocks stored on this medium.
	// Used when WeightByPrefixTokens is empty, and to choose the best
	// tier when a block is present on more than one medium.
	Weight float64 `json:"weight"`
	// WeightByPrefixTokens, when set, replaces Weight for scoring. Lookup
	// uses the consecutive matched prefix length on this tier, in tokens
	// (block count times blockSizeTokens). The first bucket whose
	// MaxTokens is 0 or >= that length wins; buckets are ordered by
	// increasing MaxTokens with the catch-all last.
	WeightByPrefixTokens []PrefixTokenWeight `json:"weightByPrefixTokens,omitempty"`
}

func DefaultKVCacheBackendConfig() []*KVCacheBackendConfig {
	return []*KVCacheBackendConfig{
		{Name: "gpu", Weight: 1.0},
		{Name: "cpu", Weight: 0.8},
	}
}

func normalizePrefixTokenWeights(medium string, in []PrefixTokenWeight) ([]PrefixTokenWeight, error) {
	if len(in) == 0 {
		return nil, nil
	}
	seen := make(map[int]struct{}, len(in))
	bounded := make([]PrefixTokenWeight, 0, len(in))
	var catchAll *PrefixTokenWeight
	for _, b := range in {
		if b.MaxTokens < 0 {
			return nil, fmt.Errorf("kv cache backend %q: maxTokens must be >= 0, got %d", medium, b.MaxTokens)
		}
		if b.Weight < 0 {
			return nil, fmt.Errorf("kv cache backend %q: weight must be >= 0, got %v", medium, b.Weight)
		}
		if _, dup := seen[b.MaxTokens]; dup {
			return nil, fmt.Errorf("kv cache backend %q: duplicate maxTokens %d", medium, b.MaxTokens)
		}
		seen[b.MaxTokens] = struct{}{}
		if b.MaxTokens == 0 {
			cp := b
			catchAll = &cp
			continue
		}
		bounded = append(bounded, b)
	}
	sort.Slice(bounded, func(i, j int) bool {
		return bounded[i].MaxTokens < bounded[j].MaxTokens
	})
	if catchAll != nil {
		bounded = append(bounded, *catchAll)
	}
	return bounded, nil
}

func lookupPrefixTokenWeight(buckets []PrefixTokenWeight, tokens int, fallback float64) float64 {
	if len(buckets) == 0 {
		return fallback
	}
	for _, b := range buckets {
		if b.MaxTokens == 0 || tokens <= b.MaxTokens {
			return b.Weight
		}
	}
	return fallback
}

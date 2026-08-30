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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

const (
	testModelName = "test-model"
	podA          = "pod-a"
	podB          = "pod-b"
)

// TestLongestPrefixScorer verifies scoring based on consecutive block hits from the start.
func TestLongestPrefixScorer(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {
			{PodIdentifier: podA, DeviceTier: "gpu"},
			{PodIdentifier: podA, DeviceTier: "cpu"},
		},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 3.0,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func TestLongestPrefixScorerDifferentTiers(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {{PodIdentifier: podA, DeviceTier: "cpu"}},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 2.5,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func TestLongestPrefixScorerWeightByPrefixTokens(t *testing.T) {
	// 4 CPU blocks * 16 tokens = 64 tokens -> 0.72 bucket.
	// 10 CPU blocks * 16 = 160 tokens -> still 0.72.
	// 128 CPU blocks * 16 = 2048 tokens -> 0.80 bucket.
	scorer, err := kvcache.NewKVBlockScorer(&kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.LongestPrefixMatch,
		BlockSizeTokens: 16,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{Name: "gpu", Weight: 1.0},
			{
				Name:   "cpu",
				Weight: 0.8,
				WeightByPrefixTokens: []kvcache.PrefixTokenWeight{
					{MaxTokens: 1024, Weight: 0.72},
					{MaxTokens: 2048, Weight: 0.80},
					{MaxTokens: 0, Weight: 0.98},
				},
			},
		},
	})
	assert.NoError(t, err)

	t.Run("short cpu prefix uses first bucket", func(t *testing.T) {
		keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3, 4})
		hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
			1: {{PodIdentifier: podA, DeviceTier: "cpu"}},
			2: {{PodIdentifier: podA, DeviceTier: "cpu"}},
			3: {{PodIdentifier: podA, DeviceTier: "cpu"}},
			4: {{PodIdentifier: podA, DeviceTier: "cpu"}},
		}
		scored, err := scorer.Score(context.Background(), keys, hitmap)
		assert.NoError(t, err)
		assert.InDelta(t, 4*0.72, scored[podA], 1e-9)
	})

	t.Run("mixed prefix looks up cpu weight from cpu tokens only", func(t *testing.T) {
		// 6 GPU + 4 CPU. CPU tokens = 64, so w_CPU=0.72, not the catch-all.
		keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
		hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{}
		for i, k := range keys {
			tier := "gpu"
			if i >= 6 {
				tier = "cpu"
			}
			hitmap[k] = []kvblock.PodEntry{{PodIdentifier: podA, DeviceTier: tier}}
		}
		scored, err := scorer.Score(context.Background(), keys, hitmap)
		assert.NoError(t, err)
		assert.InDelta(t, 6*1.0+4*0.72, scored[podA], 1e-9)
	})

	t.Run("long cpu prefix uses catch-all", func(t *testing.T) {
		n := 130 // 130*16 = 2080 tokens > 2048
		keys := make([]kvblock.BlockHash, n)
		hitmap := make(map[kvblock.BlockHash][]kvblock.PodEntry, n)
		for i := range n {
			h := kvblock.BlockHash(uint64(i + 1))
			keys[i] = h
			hitmap[h] = []kvblock.PodEntry{{PodIdentifier: podA, DeviceTier: "cpu"}}
		}
		scored, err := scorer.Score(context.Background(), keys, hitmap)
		assert.NoError(t, err)
		assert.InDelta(t, float64(n)*0.98, scored[podA], 1e-9)
	})

	t.Run("gpu+cpu same block prefers gpu", func(t *testing.T) {
		keys := int64KeysToKVBlockKeys([]uint64{1})
		hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
			1: {
				{PodIdentifier: podA, DeviceTier: "cpu"},
				{PodIdentifier: podA, DeviceTier: "gpu"},
			},
		}
		scored, err := scorer.Score(context.Background(), keys, hitmap)
		assert.NoError(t, err)
		assert.InDelta(t, 1.0, scored[podA], 1e-9)
	})
}

func TestNewKVBlockScorerRejectsDuplicateBuckets(t *testing.T) {
	_, err := kvcache.NewKVBlockScorer(&kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.LongestPrefixMatch,
		BackendConfigs: []*kvcache.KVCacheBackendConfig{
			{
				Name:   "cpu",
				Weight: 0.8,
				WeightByPrefixTokens: []kvcache.PrefixTokenWeight{
					{MaxTokens: 1024, Weight: 0.7},
					{MaxTokens: 1024, Weight: 0.8},
				},
			},
		},
	})
	assert.Error(t, err)
}

func int64KeysToKVBlockKeys(keys []uint64) []kvblock.BlockHash {
	kvKeys := make([]kvblock.BlockHash, len(keys))
	for i, key := range keys {
		kvKeys[i] = kvblock.BlockHash(key)
	}
	return kvKeys
}

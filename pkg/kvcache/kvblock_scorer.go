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
	"context"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
)

const defaultBlockSizeTokens = 16

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
	// BlockSizeTokens converts per-tier block counts to tokens for
	// WeightByPrefixTokens lookup. Zero uses defaultBlockSizeTokens.
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
		BackendConfigs:  DefaultKVCacheBackendConfig(),
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		weightMap := make(map[string]float64)
		bucketMap := make(map[string][]PrefixTokenWeight)
		for _, medium := range config.BackendConfigs {
			if medium == nil {
				continue
			}
			weightMap[medium.Name] = medium.Weight
			if len(medium.WeightByPrefixTokens) == 0 {
				continue
			}
			normalized, err := normalizePrefixTokenWeights(medium.Name, medium.WeightByPrefixTokens)
			if err != nil {
				return nil, err
			}
			bucketMap[medium.Name] = normalized
		}

		return &LongestPrefixScorer{
			MediumWeights:      weightMap,
			PrefixTokenWeights: bucketMap,
			BlockSizeTokens:    config.BlockSizeTokens,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct {
	// MediumWeights maps medium/device tier names to their scoring weights.
	// Also used to pick the best tier when a block is stored on more than one.
	MediumWeights map[string]float64
	// PrefixTokenWeights, when set for a tier, replaces MediumWeights for
	// scoring that tier. Lookup is once per consecutive prefix, from the
	// matched token count on that tier.
	PrefixTokenWeights map[string][]PrefixTokenWeight
	// BlockSizeTokens is tokens per KV block. Used only with PrefixTokenWeights.
	BlockSizeTokens int
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

func (s *LongestPrefixScorer) blockSize() int {
	if s.BlockSizeTokens > 0 {
		return s.BlockSizeTokens
	}
	return defaultBlockSizeTokens
}

func (s *LongestPrefixScorer) scalarWeight(medium string) float64 {
	if s.MediumWeights != nil {
		if w, exists := s.MediumWeights[medium]; exists {
			return w
		}
	}
	return 1.0
}

func (s *LongestPrefixScorer) weightFor(medium string, tokens int) float64 {
	fallback := s.scalarWeight(medium)
	if s.PrefixTokenWeights == nil {
		return fallback
	}
	buckets, ok := s.PrefixTokenWeights[medium]
	if !ok || len(buckets) == 0 {
		return fallback
	}
	return lookupPrefixTokenWeight(buckets, tokens, fallback)
}

// fillBestTier populates dstWeight and dstTier with the highest-scalar-weight
// device tier per podID for the given entries. The caller must clear both
// maps before calling.
func fillBestTier(dstWeight map[string]float64, dstTier map[string]string, entries []kvblock.PodEntry, mediumWeights map[string]float64) {
	for _, entry := range entries {
		weight := 1.0
		if mediumWeights != nil {
			if w, exists := mediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := dstWeight[entry.PodIdentifier]; !exists || weight > cur {
			dstWeight[entry.PodIdentifier] = weight
			dstTier[entry.PodIdentifier] = entry.DeviceTier
		}
	}
}

// Score implements the longest prefix scoring logic. Consecutive hits on a
// pod are grouped by device tier; each tier is then weighted once from its
// matched token count.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	curWeights := make(map[string]float64)
	curTiers := make(map[string]string)
	fillBestTier(curWeights, curTiers, keyToPods[keys[0]], s.MediumWeights)

	activePods := make(map[string]struct{}, len(curTiers))
	tierCounts := make(map[string]map[string]int, len(curTiers))
	for pod, tier := range curTiers {
		activePods[pod] = struct{}{}
		tierCounts[pod] = map[string]int{tier: 1}
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		clear(curTiers)
		fillBestTier(curWeights, curTiers, keyToPods[keys[i]], s.MediumWeights)

		for pod := range activePods {
			tier, exists := curTiers[pod]
			if !exists {
				delete(activePods, pod)
				continue
			}
			tierCounts[pod][tier]++
		}
	}

	blockSize := s.blockSize()
	podScores := make(map[string]float64, len(tierCounts))
	for pod, counts := range tierCounts {
		var score float64
		for tier, n := range counts {
			score += float64(n) * s.weightFor(tier, n*blockSize)
		}
		podScores[pod] = score
	}
	return podScores, nil
}

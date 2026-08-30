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

package kvcache

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLookupPrefixTokenWeight(t *testing.T) {
	buckets, err := normalizePrefixTokenWeights("cpu", []PrefixTokenWeight{
		{MaxTokens: 2048, Weight: 0.80},
		{MaxTokens: 0, Weight: 0.98},
		{MaxTokens: 1024, Weight: 0.72},
	})
	require.NoError(t, err)
	require.Equal(t, []PrefixTokenWeight{
		{MaxTokens: 1024, Weight: 0.72},
		{MaxTokens: 2048, Weight: 0.80},
		{MaxTokens: 0, Weight: 0.98},
	}, buckets)

	assert.InDelta(t, 0.72, lookupPrefixTokenWeight(buckets, 16, 0.8), 1e-9)
	assert.InDelta(t, 0.72, lookupPrefixTokenWeight(buckets, 1024, 0.8), 1e-9)
	assert.InDelta(t, 0.80, lookupPrefixTokenWeight(buckets, 1025, 0.8), 1e-9)
	assert.InDelta(t, 0.80, lookupPrefixTokenWeight(buckets, 2048, 0.8), 1e-9)
	assert.InDelta(t, 0.98, lookupPrefixTokenWeight(buckets, 2049, 0.8), 1e-9)
	assert.InDelta(t, 0.8, lookupPrefixTokenWeight(nil, 2048, 0.8), 1e-9)
}

func TestNormalizePrefixTokenWeightsRejectsInvalid(t *testing.T) {
	_, err := normalizePrefixTokenWeights("cpu", []PrefixTokenWeight{
		{MaxTokens: -1, Weight: 0.5},
	})
	require.Error(t, err)

	_, err = normalizePrefixTokenWeights("cpu", []PrefixTokenWeight{
		{MaxTokens: 1024, Weight: -0.1},
	})
	require.Error(t, err)

	_, err = normalizePrefixTokenWeights("cpu", []PrefixTokenWeight{
		{MaxTokens: 1024, Weight: 0.7},
		{MaxTokens: 1024, Weight: 0.8},
	})
	require.Error(t, err)
}

func TestKVCacheBackendConfigJSON(t *testing.T) {
	raw := []byte(`{
		"name": "cpu",
		"weight": 0.8,
		"weightByPrefixTokens": [
			{"maxTokens": 1024, "weight": 0.72},
			{"maxTokens": 0, "weight": 0.98}
		]
	}`)
	var got KVCacheBackendConfig
	require.NoError(t, json.Unmarshal(raw, &got))
	assert.Equal(t, "cpu", got.Name)
	assert.InDelta(t, 0.8, got.Weight, 1e-9)
	require.Len(t, got.WeightByPrefixTokens, 2)
	assert.Equal(t, 1024, got.WeightByPrefixTokens[0].MaxTokens)
	assert.InDelta(t, 0.72, got.WeightByPrefixTokens[0].Weight, 1e-9)
	assert.Equal(t, 0, got.WeightByPrefixTokens[1].MaxTokens)
	assert.InDelta(t, 0.98, got.WeightByPrefixTokens[1].Weight, 1e-9)
}

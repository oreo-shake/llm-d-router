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

package slaaware

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestTierTableNormalize(t *testing.T) {
	tbl := TierTable{
		Tiers: []Tier{{Name: "gold", TTFTSLOMs: 200, TPOTSLOMs: 50}},
	}
	require.NoError(t, tbl.Normalize())
	assert.InDelta(t, defaultEpsilon, tbl.DefaultEpsilonTTFT, 1e-9)
	assert.InDelta(t, 1.0, tbl.Tiers[0].Weight, 1e-9)
	assert.InDelta(t, defaultEpsilon, tbl.Tiers[0].EpsilonTTFT, 1e-9)
}

func TestResolveRequestSLO_TierHeader(t *testing.T) {
	tbl := TierTable{
		Tiers: []Tier{{
			Name: "gold", TTFTSLOMs: 200, TPOTSLOMs: 40,
			EpsilonTTFT: 0.05, EpsilonTPOT: 0.1, Weight: 2,
		}},
	}
	require.NoError(t, tbl.Normalize())
	slo, err := tbl.ResolveRequestSLO(map[string]string{TierHeaderKey: "gold"})
	require.NoError(t, err)
	assert.Equal(t, "gold", slo.TierName)
	assert.InDelta(t, 200, slo.TTFTSLOMs, 1e-9)
	assert.InDelta(t, 0.05, slo.EpsilonTTFT, 1e-9)
}

func TestResolveRequestSLO_HeaderSLOs(t *testing.T) {
	tbl := TierTable{}
	require.NoError(t, tbl.Normalize())
	slo, err := tbl.ResolveRequestSLO(map[string]string{
		metadata.TTFTSLOHeaderKey: "150",
		metadata.TPOTSLOHeaderKey: "30",
	})
	require.NoError(t, err)
	assert.InDelta(t, 150, slo.TTFTSLOMs, 1e-9)
	assert.InDelta(t, 30, slo.TPOTSLOMs, 1e-9)
}

func TestResolveRequestSLO_UnknownTier(t *testing.T) {
	tbl := TierTable{Tiers: []Tier{{Name: "gold", TTFTSLOMs: 100}}}
	require.NoError(t, tbl.Normalize())
	_, err := tbl.ResolveRequestSLO(map[string]string{TierHeaderKey: "silver"})
	assert.Error(t, err)
}

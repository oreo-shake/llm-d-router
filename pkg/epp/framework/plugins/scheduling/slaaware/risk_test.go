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
)

func TestQuantileFeasible(t *testing.T) {
	slo := RequestSLO{TTFTSLOMs: 200, TPOTSLOMs: 50}
	assert.True(t, QuantileFeasible(100, 40, slo))
	assert.False(t, QuantileFeasible(250, 40, slo))
	assert.False(t, QuantileFeasible(100, 60, slo))
}

func TestSurvivalFromQuantile_Monotone(t *testing.T) {
	low := SurvivalFromQuantile(50, 200, 0.25)
	high := SurvivalFromQuantile(250, 200, 0.25)
	assert.Less(t, low, 0.5)
	assert.Greater(t, high, 0.5)
	assert.Less(t, low, high)
}

func TestCostNowAndViolation(t *testing.T) {
	slo := RequestSLO{TTFTSLOMs: 100, TPOTSLOMs: 20, AlphaTTFT: 1, AlphaTPOT: 1, Scale: 0.25}
	ok := RiskFromQuantile(50, 10, slo)
	bad := RiskFromQuantile(200, 40, slo)
	assert.True(t, ok.Feasible)
	assert.False(t, bad.Feasible)
	assert.Less(t, CostNow(ok, slo), CostNow(bad, slo))
	assert.Greater(t, CostViolation(bad, slo), 0.0)
	assert.InDelta(t, 0.0, CostViolation(ok, slo), 1e-9)
}

func TestRiskFromCDF_EpsilonBudget(t *testing.T) {
	slo := RequestSLO{TTFTSLOMs: 100, TPOTSLOMs: 20, EpsilonTTFT: 0.1, EpsilonTPOT: 0.1}
	ok := RiskFromCDF(0.05, 0.05, 80, 15, slo)
	bad := RiskFromCDF(0.2, 0.05, 120, 15, slo)
	assert.True(t, ok.Feasible)
	assert.False(t, bad.Feasible)
	assert.Equal(t, "cdf", ok.Source)
}

func TestScoreFromCost(t *testing.T) {
	assert.InDelta(t, 1.0, ScoreFromCost(0), 1e-9)
	assert.Greater(t, ScoreFromCost(0), ScoreFromCost(2))
	assert.Equal(t, 0.0, ScoreFromCost(-1))
}

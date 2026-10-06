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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
)

func TestDistAFTSurvivalCentered(t *testing.T) {
	m := newDistAFT()
	x := make([]float64, aftFeatN)
	x[0] = 1
	for i := 0; i < 80; i++ {
		m.Observe(x, 400)
	}
	mid := m.Survival(x, 400)
	assert.InDelta(t, 0.5, mid, 0.15)
	assert.Greater(t, m.Survival(x, 100), m.Survival(x, 800))
	assert.Greater(t, m.ExpectedOverrun(x, 100), m.ExpectedOverrun(x, 800))
	assert.False(t, math.IsNaN(mid))
}

func TestDistMLPColdStartMatchesSoftWrap(t *testing.T) {
	m := newDistMLP()
	slo := 1000.0
	// High L-hat vs S → soft wrap near 1; skip connection dominates at n=0.
	xHi := exceedFeatures(latencypredictor.LatencyShape{}, nil, 100, 0.5, 0, 0, 5000, slo)
	xLo := exceedFeatures(latencypredictor.LatencyShape{}, nil, 100, 0.5, 0, 0, 100, slo)
	pHi := m.Predict(xHi, slo)
	pLo := m.Predict(xLo, slo)
	assert.Greater(t, pHi, 0.85)
	assert.Less(t, pLo, 0.15)
	assert.Greater(t, pHi, pLo)
}

func TestDistMLPMonotoneInSAndLearns(t *testing.T) {
	m := newDistMLP()
	// Same L-hat; tighter S should yield higher P(L>S).
	pred := 400.0
	xGold := exceedFeatures(latencypredictor.LatencyShape{}, nil, 100, 0.2, 0, 0, pred, 200)
	xBronze := exceedFeatures(latencypredictor.LatencyShape{}, nil, 100, 0.2, 0, 0, pred, 2000)
	assert.Greater(t, m.Predict(xGold, 200), m.Predict(xBronze, 2000))

	for i := 0; i < 60; i++ {
		// Miss gold (actual 400 > 200), hit bronze (400 < 2000).
		m.Observe(xGold, 200, 400)
		m.Observe(xBronze, 2000, 400)
	}
	assert.True(t, m.ready())
	pGold := m.Predict(xGold, 200)
	pBronze := m.Predict(xBronze, 2000)
	assert.Greater(t, pGold, pBronze)
	assert.Greater(t, math.Abs(pGold-0.5), 0.05)
}

func TestDistMLPSeparatesMissUsingLhat(t *testing.T) {
	m := newDistMLP()
	slo := 3850.0
	// Train: when L-hat >> S, actual often misses; when L-hat << S, hits.
	for i := 0; i < 40; i++ {
		xMiss := exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.1, 1000, 0, 12000, slo)
		xHit := exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.8, 0, 0, 800, slo)
		m.Observe(xMiss, slo, 15000)
		m.Observe(xHit, slo, 2000)
	}
	pMiss := m.Predict(exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.1, 1000, 0, 12000, slo), slo)
	pHit := m.Predict(exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.8, 0, 0, 800, slo), slo)
	require.Greater(t, pMiss, pHit)
	assert.Greater(t, pMiss-pHit, 0.2)
}

func TestExceedFeaturesPacksMargin(t *testing.T) {
	x := exceedFeatures(latencypredictor.LatencyShape{UnmatchedTokens: 100, ActiveReloadJobs: 2}, nil, 1000, 0.5, 10, 20, 2000, 1000)
	require.Len(t, x, excFeatN)
	assert.InDelta(t, 1.0, x[19], 1e-9) // (2000-1000)/1000
	assert.InDelta(t, math.Log(1000), x[21], 1e-9)
	assert.InDelta(t, 2.0, x[22], 1e-9)
}

func TestDistMLPResidualBoundedKeepsSafeLow(t *testing.T) {
	m := newDistMLP()
	slo := 12800.0
	// Many gold-style misses with high L-hat should not push safe silver
	// (L-hat << S) into high p — residual is tanh-bounded.
	for i := 0; i < 80; i++ {
		xMiss := exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.1, 2000, 0, 20000, 3850)
		m.Observe(xMiss, 3850, 25000)
	}
	xSafe := exceedFeatures(latencypredictor.LatencyShape{}, nil, 8000, 0.8, 0, 0, 2000, slo)
	pSafe := m.Predict(xSafe, slo)
	assert.Less(t, pSafe, 0.25, "safe L-hat<<S must stay low after miss training")
}

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
}

func TestDistLogisticMonotoneInS(t *testing.T) {
	m := newDistLogistic()
	x := make([]float64, aftFeatN)
	x[0] = 1
	for i := 0; i < 40; i++ {
		m.Observe(x, 200, 400)  // miss gold
		m.Observe(x, 2000, 400) // hit bronze
	}
	assert.Greater(t, m.Predict(x, 200), m.Predict(x, 2000))
	assert.True(t, m.ready())
	assert.Greater(t, math.Abs(m.Predict(x, 200)-0.5), 0.05)
}

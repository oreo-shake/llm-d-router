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
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
)

const (
	distMinSamples = 12
	distLR         = 0.05
	distL2         = 1e-4
	distMinSigma   = 0.25
	distWarmSigma  = 0.8
	aftFeatN       = 12
)

// distAFT is an online lognormal AFT: log L ~ N(w·x, sigma^2) with sigma
// from the residual of that fit. x does not include S; S is applied at
// Survival time so any SLO works.
type distAFT struct {
	mu    sync.Mutex
	w     [aftFeatN]float64
	n     int
	rMean float64
	rEx2  float64
}

func newDistAFT() *distAFT {
	return &distAFT{}
}

func (m *distAFT) Observe(x []float64, latencyMs float64) {
	if m == nil || latencyMs <= 0 || len(x) < aftFeatN {
		return
	}
	y := math.Log(latencyMs)
	m.mu.Lock()
	defer m.mu.Unlock()
	pred := dotN(m.w[:], x)
	err := pred - y
	for i := 0; i < aftFeatN; i++ {
		m.w[i] -= distLR * (err*x[i] + distL2*m.w[i])
	}
	if m.n == 0 {
		m.rMean = err
		m.rEx2 = err * err
		m.n = 1
		return
	}
	a := 0.05
	m.rMean = (1-a)*m.rMean + a*err
	m.rEx2 = (1-a)*m.rEx2 + a*err*err
	m.n++
}

func (m *distAFT) ready() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.n >= distMinSamples
}

func (m *distAFT) muSigma(x []float64) (mu, sigma float64) {
	if m == nil || len(x) < aftFeatN {
		return 0, distWarmSigma
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	mu = dotN(m.w[:], x)
	if m.n < distMinSamples {
		return mu, distWarmSigma
	}
	v := m.rEx2 - m.rMean*m.rMean
	if v < distMinSigma*distMinSigma {
		return mu, distMinSigma
	}
	return mu, math.Sqrt(v)
}

func (m *distAFT) Survival(x []float64, sloMs float64) float64 {
	if sloMs <= 0 {
		return 0
	}
	mu, sigma := m.muSigma(x)
	z := (math.Log(sloMs) - mu) / sigma
	return 1 - normalCDF(z)
}

func (m *distAFT) ExpectedOverrun(x []float64, sloMs float64) float64 {
	if sloMs <= 0 {
		return 0
	}
	mu, sigma := m.muSigma(x)
	logS := math.Log(sloMs)
	z := (logS - mu) / sigma
	surv := 1 - normalCDF(z)
	mean := math.Exp(mu + 0.5*sigma*sigma)
	z2 := (mu + sigma*sigma - logS) / sigma
	return math.Max(0, mean*normalCDF(z2)-sloMs*surv)
}

// distLogistic is online P(L > S | phi, log S) via SGD log-loss.
type distLogistic struct {
	mu sync.Mutex
	w  [aftFeatN + 1]float64
	n  int
}

func newDistLogistic() *distLogistic {
	return &distLogistic{}
}

func (m *distLogistic) Observe(x []float64, sloMs, latencyMs float64) {
	if m == nil || sloMs <= 0 || latencyMs <= 0 || len(x) < aftFeatN {
		return
	}
	y := 0.0
	if latencyMs > sloMs {
		y = 1
	}
	xs := appendLogS(x, sloMs)
	m.mu.Lock()
	defer m.mu.Unlock()
	p := sigmoid(dotN(m.w[:], xs))
	err := p - y
	for i := 0; i < len(m.w); i++ {
		m.w[i] -= distLR * (err*xs[i] + distL2*m.w[i])
	}
	m.n++
}

func (m *distLogistic) ready() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.n >= distMinSamples
}

func (m *distLogistic) Predict(x []float64, sloMs float64) float64 {
	if m == nil || sloMs <= 0 || len(x) < aftFeatN {
		return 0.5
	}
	xs := appendLogS(x, sloMs)
	m.mu.Lock()
	defer m.mu.Unlock()
	return sigmoid(dotN(m.w[:], xs))
}

func appendLogS(x []float64, sloMs float64) []float64 {
	out := make([]float64, aftFeatN+1)
	copy(out, x[:aftFeatN])
	out[aftFeatN] = math.Log(sloMs)
	return out
}

func distFeatures(s latencypredictor.LatencyShape, metrics *fwkdl.Metrics, inputTokens int) []float64 {
	kv := 0.0
	queue := 0.0
	running := 0.0
	if metrics != nil {
		kv = metrics.KVCacheUsagePercent
		queue = float64(metrics.WaitingQueueSize)
		running = float64(metrics.RunningRequestsSize)
	}
	x := make([]float64, aftFeatN)
	x[0] = 1
	x[1] = kv
	x[2] = math.Log1p(float64(inputTokens)) / 10
	x[3] = float64(inputTokens) / 10000
	x[4] = queue / 32
	x[5] = running / 32
	x[6] = s.KVFreeFraction
	x[7] = float64(s.UnmatchedTokens) / 10000
	x[8] = float64(s.WaitingTokens) / 1e5
	x[9] = s.EwmaTTFTMs / 5000
	x[10] = float64(s.UncachedRequestTokens) / 10000
	x[11] = s.LogInputTokenLength / 10
	return x
}

func dotN(w, x []float64) float64 {
	n := len(w)
	if len(x) < n {
		n = len(x)
	}
	s := 0.0
	for i := 0; i < n; i++ {
		s += w[i] * x[i]
	}
	return s
}

func sigmoid(z float64) float64 {
	if z > 20 {
		return 1
	}
	if z < -20 {
		return 0
	}
	return 1 / (1 + math.Exp(-z))
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

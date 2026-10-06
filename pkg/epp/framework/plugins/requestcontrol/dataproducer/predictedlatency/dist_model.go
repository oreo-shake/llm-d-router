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

	// Exceedance MLP: rich phi + L-hat margin, local matmuls only.
	// logit = softWrapLogit + mlpResidAlpha * tanh(MLP); soft wrap is the
	// base; the net learns a bounded residual so rare-positive SGD cannot
	// drive mean p far above the miss rate.
	excFeatN      = 23
	mlpH1         = 32
	mlpH2         = 16
	mlpLR         = 0.01
	mlpL2         = 1e-4
	mlpPosWeight  = 3.0
	mlpResidAlpha = 2.0
	softScale     = 0.25
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
	if sigma <= 0 || math.IsNaN(mu) || math.IsInf(mu, 0) {
		return 0.5
	}
	z := (math.Log(sloMs) - mu) / sigma
	p := 1 - normalCDF(z)
	if math.IsNaN(p) {
		return 0.5
	}
	return p
}

func (m *distAFT) ExpectedOverrun(x []float64, sloMs float64) float64 {
	if sloMs <= 0 {
		return 0
	}
	mu, sigma := m.muSigma(x)
	if sigma <= 0 || math.IsNaN(mu) || math.IsInf(mu, 0) {
		return 0
	}
	logS := math.Log(sloMs)
	z := (logS - mu) / sigma
	surv := 1 - normalCDF(z)
	mean := math.Exp(mu + 0.5*sigma*sigma)
	z2 := (mu + sigma*sigma - logS) / sigma
	return math.Max(0, mean*normalCDF(z2)-sloMs*surv)
}

// distMLP is online P(L>S | phi, L-hat, S) via a tiny 2-hidden MLP.
// logit = softWrapLogit(x) + mlpResidAlpha * tanh(MLP(x)). Soft wrap is
// frozen as the base; MLP adds a bounded residual in (-alpha, +alpha).
type distMLP struct {
	mu sync.Mutex

	w1 [mlpH1][excFeatN]float64
	b1 [mlpH1]float64
	w2 [mlpH2][mlpH1]float64
	b2 [mlpH2]float64
	w3 [mlpH2]float64
	b3 float64

	n int
}

func newDistMLP() *distMLP {
	m := &distMLP{}
	// Small He-style init; cold start ≈ soft wrap because tanh(MLP)≈0.
	scale1 := math.Sqrt(2 / float64(excFeatN))
	scale2 := math.Sqrt(2 / float64(mlpH1))
	scale3 := math.Sqrt(2 / float64(mlpH2))
	seed := 1.0
	next := func() float64 {
		seed = math.Mod(seed*1.6180339887+0.1, 1)
		return seed - 0.5
	}
	for i := 0; i < mlpH1; i++ {
		for j := 0; j < excFeatN; j++ {
			m.w1[i][j] = next() * scale1 * 0.1
		}
	}
	for i := 0; i < mlpH2; i++ {
		for j := 0; j < mlpH1; j++ {
			m.w2[i][j] = next() * scale2 * 0.1
		}
	}
	for i := 0; i < mlpH2; i++ {
		m.w3[i] = next() * scale3 * 0.1
	}
	return m
}

// newDistLogistic keeps the old constructor name used by plugin wiring.
func newDistLogistic() *distMLP {
	return newDistMLP()
}

func (m *distMLP) ready() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.n >= distMinSamples
}

func (m *distMLP) Predict(x []float64, sloMs float64) float64 {
	if m == nil || sloMs <= 0 || len(x) < excFeatN {
		return 0.5
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	_, _, _, p := m.forwardLocked(x)
	return p
}

func (m *distMLP) Observe(x []float64, sloMs, latencyMs float64) {
	if m == nil || sloMs <= 0 || latencyMs <= 0 || len(x) < excFeatN {
		return
	}
	y := 0.0
	if latencyMs > sloMs {
		y = 1
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	h1, h2, raw, p := m.forwardLocked(x)
	wy := 1.0
	if y == 1 {
		wy = mlpPosWeight
	}
	// dL/dz = w (p - y); z = softZ + alpha * tanh(raw)
	gz := wy * (p - y)
	dtanh := 1 - math.Tanh(raw)*math.Tanh(raw)
	graw := gz * mlpResidAlpha * dtanh

	// output layer (into raw)
	gw3 := make([]float64, mlpH2)
	gh2 := make([]float64, mlpH2)
	for j := 0; j < mlpH2; j++ {
		gw3[j] = graw * h2[j]
		gh2[j] = graw * m.w3[j]
		if h2[j] <= 0 {
			gh2[j] = 0
		}
	}
	for j := 0; j < mlpH2; j++ {
		m.w3[j] -= mlpLR * (gw3[j] + mlpL2*m.w3[j])
	}
	m.b3 -= mlpLR * graw

	// hidden 2
	gw2 := [mlpH2][mlpH1]float64{}
	gh1 := make([]float64, mlpH1)
	for i := 0; i < mlpH2; i++ {
		for j := 0; j < mlpH1; j++ {
			gw2[i][j] = gh2[i] * h1[j]
			gh1[j] += gh2[i] * m.w2[i][j]
		}
	}
	for j := 0; j < mlpH1; j++ {
		if h1[j] <= 0 {
			gh1[j] = 0
		}
	}
	for i := 0; i < mlpH2; i++ {
		for j := 0; j < mlpH1; j++ {
			m.w2[i][j] -= mlpLR * (gw2[i][j] + mlpL2*m.w2[i][j])
		}
		m.b2[i] -= mlpLR * gh2[i]
	}

	// hidden 1
	for i := 0; i < mlpH1; i++ {
		for j := 0; j < excFeatN; j++ {
			m.w1[i][j] -= mlpLR * (gh1[i]*x[j] + mlpL2*m.w1[i][j])
		}
		m.b1[i] -= mlpLR * gh1[i]
	}

	m.n++
}

// forwardLocked returns h1, h2, MLP raw pre-tanh activation, and p.
// logit = softWrapLogit + mlpResidAlpha * tanh(raw).
func (m *distMLP) forwardLocked(x []float64) (h1 [mlpH1]float64, h2 [mlpH2]float64, raw, p float64) {
	for i := 0; i < mlpH1; i++ {
		s := m.b1[i]
		for j := 0; j < excFeatN; j++ {
			s += m.w1[i][j] * x[j]
		}
		if s > 0 {
			h1[i] = s
		}
	}
	for i := 0; i < mlpH2; i++ {
		s := m.b2[i]
		for j := 0; j < mlpH1; j++ {
			s += m.w2[i][j] * h1[j]
		}
		if s > 0 {
			h2[i] = s
		}
	}
	raw = m.b3
	for j := 0; j < mlpH2; j++ {
		raw += m.w3[j] * h2[j]
	}
	z := softWrapLogit(x) + mlpResidAlpha*math.Tanh(raw)
	p = sigmoid(z)
	return h1, h2, raw, p
}

// softWrapLogit reads pred/S features packed by exceedFeatures.
func softWrapLogit(x []float64) float64 {
	if len(x) < excFeatN {
		return 0
	}
	// x[19] = (pred-S)/S, soft wrap uses (pred-S)/(0.25 S) = 4 * x[19]
	return x[19] / softScale
}

// distFeatures builds the AFT feature vector (no S, no L-hat).
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

// exceedFeatures builds the MLP input: rich phi + L-hat margin + log S.
func exceedFeatures(
	s latencypredictor.LatencyShape,
	metrics *fwkdl.Metrics,
	inputTokens int,
	prefixScore float64,
	prefillInFlight, decodeInFlight int64,
	predMs, sloMs float64,
) []float64 {
	kv := 0.0
	queue := 0.0
	running := 0.0
	if metrics != nil {
		kv = metrics.KVCacheUsagePercent
		queue = float64(metrics.WaitingQueueSize)
		running = float64(metrics.RunningRequestsSize)
	}
	if predMs < 0 {
		predMs = 0
	}
	if sloMs <= 0 {
		sloMs = 1
	}
	margin := (predMs - sloMs) / sloMs
	x := make([]float64, excFeatN)
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
	x[12] = prefixScore
	x[13] = float64(prefillInFlight) / 1e5
	x[14] = float64(decodeInFlight) / 1e5
	x[15] = s.UnmatchedQuadraticBlocks / 1e6
	x[16] = float64(s.CachedGPUTokens) / 10000
	x[17] = float64(s.CachedCPUTokens) / 10000
	x[18] = s.EwmaTPOTMs / 50
	x[19] = margin
	x[20] = math.Log1p(predMs) / 10
	x[21] = math.Log(sloMs)
	x[22] = s.ActiveReloadJobs
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

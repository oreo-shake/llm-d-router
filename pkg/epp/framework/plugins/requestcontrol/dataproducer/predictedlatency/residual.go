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
)

const (
	residualAlpha      = 0.05
	residualMinSamples = 8
	residualMinSigma   = 0.3
	residualWarmSigma  = 1.0
	minPositiveMs      = 1.0
)

// residualTracker keeps an online log-residual (log actual - log pred) for
// TTFT and TPOT, plus per-pod EWMAs of realized latency. Used to form
// P(L > S) without waiting for a distributional sidecar.
type residualTracker struct {
	mu   sync.Mutex
	ttft logMoments
	tpot logMoments
	pods map[string]*podEWMA
}

type logMoments struct {
	n    int
	mean float64
	ex2  float64
}

type podEWMA struct {
	ttft float64
	tpot float64
	n    int
}

func newResidualTracker() *residualTracker {
	return &residualTracker{pods: make(map[string]*podEWMA)}
}

func (t *residualTracker) ObserveTTFT(pod string, pred, actual float64) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ttft.observe(pred, actual)
	p := t.podLocked(pod)
	p.ttft = ewma(p.ttft, actual, p.n == 0)
	p.n++
}

func (t *residualTracker) ObserveTPOT(pod string, pred, actual float64) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.tpot.observe(pred, actual)
	p := t.podLocked(pod)
	p.tpot = ewma(p.tpot, actual, p.n == 0)
	p.n++
}

func (t *residualTracker) TTFTSigma() float64 {
	if t == nil {
		return residualWarmSigma
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.ttft.sigma()
}

func (t *residualTracker) TPOTSigma() float64 {
	if t == nil {
		return residualWarmSigma
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.tpot.sigma()
}

func (t *residualTracker) TTFTMu() float64 {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.ttft.bias()
}

func (t *residualTracker) TPOTMu() float64 {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.tpot.bias()
}

func (t *residualTracker) PodEWMA(pod string) (ttft, tpot float64) {
	if t == nil {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	p := t.pods[pod]
	if p == nil {
		return 0, 0
	}
	return p.ttft, p.tpot
}

func (t *residualTracker) podLocked(pod string) *podEWMA {
	p := t.pods[pod]
	if p == nil {
		p = &podEWMA{}
		t.pods[pod] = p
	}
	return p
}

func (m *logMoments) observe(pred, actual float64) {
	if pred <= 0 || actual <= 0 || math.IsNaN(pred) || math.IsNaN(actual) {
		return
	}
	r := math.Log(actual) - math.Log(pred)
	if m.n == 0 {
		m.mean = r
		m.ex2 = r * r
		m.n = 1
		return
	}
	m.mean = (1-residualAlpha)*m.mean + residualAlpha*r
	m.ex2 = (1-residualAlpha)*m.ex2 + residualAlpha*r*r
	m.n++
}

func (m logMoments) sigma() float64 {
	if m.n < residualMinSamples {
		return residualWarmSigma
	}
	v := m.ex2 - m.mean*m.mean
	if v < residualMinSigma*residualMinSigma {
		return residualMinSigma
	}
	return math.Sqrt(v)
}

func (m logMoments) bias() float64 {
	if m.n < residualMinSamples {
		return 0
	}
	return m.mean
}

func ewma(prev, x float64, first bool) float64 {
	if first || prev <= 0 {
		return x
	}
	return (1-residualAlpha)*prev + residualAlpha*x
}

func clampPositivePred(pred, ewmaMs float64) float64 {
	if pred > 0 && !math.IsNaN(pred) && !math.IsInf(pred, 0) {
		return pred
	}
	if ewmaMs > 0 {
		return ewmaMs
	}
	return minPositiveMs
}

// calibratePred multiplies a point prediction by exp(mu), where mu is the
// online mean of log(actual)-log(pred). Identity when mu is 0.
func calibratePred(pred, mu float64) float64 {
	if pred <= 0 || mu == 0 || math.IsNaN(mu) {
		return pred
	}
	return pred * math.Exp(mu)
}

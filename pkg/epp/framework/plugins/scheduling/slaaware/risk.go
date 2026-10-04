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

import "math"

// EndpointRisk is the per-(request, pod) risk view used by filter and scorer.
type EndpointRisk struct {
	TTFTPred      float64
	TPOTPred      float64
	TTFTSurvival  float64 // P(TTFT > S^P) ≈ 1 - F_θ_P(S^P)
	TPOTSurvival  float64 // P(TPOT > S^D) ≈ 1 - F_θ_D(S^D)
	Feasible      bool
	Source        string // "quantile" or "cdf"
	HasPrediction bool
}

// SurvivalFromLogNormal is P(L > S) when log L ~ N(log(pred), sigma^2).
// pred is a positive residual-calibrated point estimate (ms); sigma is the
// log-residual stddev. Falls back to SurvivalFromQuantile when pred or sigma
// is non-positive.
func SurvivalFromLogNormal(pred, sigma, slo, scale float64) float64 {
	if slo <= 0 {
		return 0
	}
	if pred <= 0 || sigma <= 0 || math.IsNaN(pred) || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		return SurvivalFromQuantile(pred, slo, scale)
	}
	z := (math.Log(slo) - math.Log(pred)) / sigma
	return 1 - normalCDF(z)
}

// normalCDF is Φ(x) for the standard normal (Abramowitz and Stegun 7.1.26).
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

// SurvivalFromQuantile approximates P(L > S) from a single quantile prediction Q
// via a logistic centered at Q with scale = logisticScale * max(S, 1).
func SurvivalFromQuantile(pred, slo, scale float64) float64 {
	if slo <= 0 {
		// No threshold: treat as always feasible (zero survival / zero risk).
		return 0
	}
	if scale <= 0 {
		scale = defaultScale
	}
	denom := scale * math.Max(slo, 1)
	// σ((pred - S) / denom) rises when prediction exceeds SLO.
	return 1 / (1 + math.Exp(-(pred-slo)/denom))
}

// QuantileFeasible is true when the predicted ε-quantile is within the SLO
// on every gated dimension (ungated dims with S=0 are ignored).
func QuantileFeasible(ttftPred, tpotPred float64, slo RequestSLO) bool {
	if slo.TTFTSLOMs > 0 && ttftPred > slo.TTFTSLOMs {
		return false
	}
	if slo.TPOTSLOMs > 0 && tpotPred > slo.TPOTSLOMs {
		return false
	}
	// At least one dimension must be gated, otherwise everything is feasible.
	return slo.TTFTSLOMs > 0 || slo.TPOTSLOMs > 0
}

// RiskFromQuantile builds EndpointRisk from point/quantile predictions.
func RiskFromQuantile(ttftPred, tpotPred float64, slo RequestSLO) EndpointRisk {
	ttftSurv := SurvivalFromQuantile(ttftPred, slo.TTFTSLOMs, slo.Scale)
	tpotSurv := SurvivalFromQuantile(tpotPred, slo.TPOTSLOMs, slo.Scale)
	return EndpointRisk{
		TTFTPred:      ttftPred,
		TPOTPred:      tpotPred,
		TTFTSurvival:  ttftSurv,
		TPOTSurvival:  tpotSurv,
		Feasible:      QuantileFeasible(ttftPred, tpotPred, slo),
		Source:        "quantile",
		HasPrediction: true,
	}
}

// RiskFromCDF builds EndpointRisk from explicit survival probabilities
// P(L > S). Feasible when both survivals are ≤ the tier risk budgets.
func RiskFromCDF(ttftSurv, tpotSurv, ttftPred, tpotPred float64, slo RequestSLO) EndpointRisk {
	ttftSurv = clamp01(ttftSurv)
	tpotSurv = clamp01(tpotSurv)
	feasible := true
	gated := false
	if slo.TTFTSLOMs > 0 {
		gated = true
		if ttftSurv > slo.EpsilonTTFT {
			feasible = false
		}
	}
	if slo.TPOTSLOMs > 0 {
		gated = true
		if tpotSurv > slo.EpsilonTPOT {
			feasible = false
		}
	}
	if !gated {
		feasible = false
	}
	return EndpointRisk{
		TTFTPred:      ttftPred,
		TPOTPred:      tpotPred,
		TTFTSurvival:  ttftSurv,
		TPOTSurvival:  tpotSurv,
		Feasible:      feasible,
		Source:        "cdf",
		HasPrediction: true,
	}
}

// CostNow is C_ij^now = α_P(1-F_P) + α_D(1-F_D) = α_P·surv_P + α_D·surv_D.
func CostNow(r EndpointRisk, slo RequestSLO) float64 {
	return slo.AlphaTTFT*r.TTFTSurvival + slo.AlphaTPOT*r.TPOTSurvival
}

// CostOverrun is alpha-weighted E[(L-S)+] / S from the AFT model.
func CostOverrun(ttftOverrunMs, tpotOverrunMs float64, slo RequestSLO) float64 {
	var cost float64
	if slo.TTFTSLOMs > 0 {
		cost += slo.AlphaTTFT * math.Max(0, ttftOverrunMs) / slo.TTFTSLOMs
	}
	if slo.TPOTSLOMs > 0 {
		cost += slo.AlphaTPOT * math.Max(0, tpotOverrunMs) / slo.TPOTSLOMs
	}
	return cost
}

// CostViolation is the soft relative quantile overrun used when F_i is empty.
func CostViolation(r EndpointRisk, slo RequestSLO) float64 {
	var cost float64
	if slo.TTFTSLOMs > 0 {
		cost += slo.AlphaTTFT * math.Max(0, r.TTFTPred-slo.TTFTSLOMs) / slo.TTFTSLOMs
	}
	if slo.TPOTSLOMs > 0 {
		cost += slo.AlphaTPOT * math.Max(0, r.TPOTPred-slo.TPOTSLOMs) / slo.TPOTSLOMs
	}
	return cost
}

// ScoreFromCost maps a non-negative cost onto (0,1] (lower cost → higher score).
func ScoreFromCost(cost float64) float64 {
	if cost < 0 || math.IsNaN(cost) || math.IsInf(cost, 0) {
		return 0
	}
	return 1 / (1 + cost)
}

func clamp01(v float64) float64 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

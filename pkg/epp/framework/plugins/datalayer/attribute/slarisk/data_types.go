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

package slarisk

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sladistribconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sladistribution/constants"
)

// SLARiskInfoDataKey is written by sla-distribution-producer and consumed by
// sla-risk-filter / sla-cost-scorer when CDF (or quantile-derived) survivals are present.
var SLARiskInfoDataKey = plugin.NewDataKey("SLARiskInfoDataKey", sladistribconstants.PluginType)

// SLARiskInfo carries per-endpoint SLA risk for a request.
type SLARiskInfo struct {
	ttftPred     float64
	tpotPred     float64
	ttftSurvival float64 // P(TTFT > S)
	tpotSurvival float64 // P(TPOT > S)
	feasible     bool
	source       string // "quantile" | "cdf" | "aft" | "exceedance" | "lognormal"
	futureCost   float64
	hasFuture    bool
	ttftOverrun  float64
	tpotOverrun  float64
}

// NewSLARiskInfo constructs an SLARiskInfo.
func NewSLARiskInfo(ttftPred, tpotPred, ttftSurvival, tpotSurvival float64, feasible bool, source string) *SLARiskInfo {
	return &SLARiskInfo{
		ttftPred:     ttftPred,
		tpotPred:     tpotPred,
		ttftSurvival: ttftSurvival,
		tpotSurvival: tpotSurvival,
		feasible:     feasible,
		source:       source,
	}
}

// WithFutureCost attaches V_H(X^{+ij}) for the λ·future term.
func (s *SLARiskInfo) WithFutureCost(v float64) *SLARiskInfo {
	if s == nil {
		return nil
	}
	s.futureCost = v
	s.hasFuture = true
	return s
}

func (s *SLARiskInfo) TTFTPred() float64     { return s.ttftPred }
func (s *SLARiskInfo) TPOTPred() float64     { return s.tpotPred }
func (s *SLARiskInfo) TTFTSurvival() float64 { return s.ttftSurvival }
func (s *SLARiskInfo) TPOTSurvival() float64 { return s.tpotSurvival }
func (s *SLARiskInfo) Feasible() bool        { return s.feasible }
func (s *SLARiskInfo) Source() string        { return s.source }
func (s *SLARiskInfo) FutureCost() float64   { return s.futureCost }
func (s *SLARiskInfo) HasFuture() bool       { return s.hasFuture }
func (s *SLARiskInfo) WithOverrun(ttftMs, tpotMs float64) *SLARiskInfo {
	if s == nil {
		return nil
	}
	s.ttftOverrun = ttftMs
	s.tpotOverrun = tpotMs
	return s
}

func (s *SLARiskInfo) TTFTOverrun() float64 { return s.ttftOverrun }
func (s *SLARiskInfo) TPOTOverrun() float64 { return s.tpotOverrun }

// Clone implements fwkdl.Cloneable.
func (s *SLARiskInfo) Clone() fwkdl.Cloneable {
	if s == nil {
		return nil
	}
	cp := *s
	return &cp
}

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

// Package slaaware holds shared SLA-tier contracts and risk/cost math used by
// the sla-risk-filter, sla-cost-scorer, and sla-distribution-producer plugins.
package slaaware

import (
	"fmt"
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// TierHeaderKey names the optional request header selecting a configured SLA tier.
	TierHeaderKey = "x-llm-d-slo-tier"

	defaultEpsilon = 0.1
	defaultAlphaP  = 1.0
	defaultAlphaD  = 1.0
	defaultScale   = 0.25
)

// Tier is one SLA class C with TTFT/TPOT thresholds, risk budgets, and value weight.
type Tier struct {
	Name        string  `json:"name"`
	TTFTSLOMs   float64 `json:"ttftSloMs"`
	TPOTSLOMs   float64 `json:"tpotSloMs"`
	EpsilonTTFT float64 `json:"epsilonTtft,omitempty"`
	EpsilonTPOT float64 `json:"epsilonTpot,omitempty"`
	Weight      float64 `json:"weight,omitempty"`
}

// TierTable is the plugin-facing SLA tier configuration.
type TierTable struct {
	// Tiers maps named tiers. When empty, request SLO headers alone define thresholds.
	Tiers []Tier `json:"tiers,omitempty"`
	// DefaultEpsilonTTFT is used when a tier omits epsilonTtft and no tier is selected.
	DefaultEpsilonTTFT float64 `json:"defaultEpsilonTtft,omitempty"`
	// DefaultEpsilonTPOT is used when a tier omits epsilonTpot and no tier is selected.
	DefaultEpsilonTPOT float64 `json:"defaultEpsilonTpot,omitempty"`
	// AlphaTTFT is α_P in C_now / C_violation. Default 1.
	AlphaTTFT float64 `json:"alphaTtft,omitempty"`
	// AlphaTPOT is α_D in C_now / C_violation. Default 1.
	AlphaTPOT float64 `json:"alphaTpot,omitempty"`
	// LogisticScale controls the quantile→survival map scale. Default 0.25.
	LogisticScale float64 `json:"logisticScale,omitempty"`
}

// RequestSLO is the resolved per-request SLA target used for risk checks and costs.
type RequestSLO struct {
	TierName    string
	TTFTSLOMs   float64
	TPOTSLOMs   float64
	EpsilonTTFT float64
	EpsilonTPOT float64
	Weight      float64
	AlphaTTFT   float64
	AlphaTPOT   float64
	Scale       float64
}

// Normalize fills defaults and validates a TierTable.
func (t *TierTable) Normalize() error {
	if t.DefaultEpsilonTTFT == 0 {
		t.DefaultEpsilonTTFT = defaultEpsilon
	}
	if t.DefaultEpsilonTPOT == 0 {
		t.DefaultEpsilonTPOT = defaultEpsilon
	}
	if t.AlphaTTFT == 0 {
		t.AlphaTTFT = defaultAlphaP
	}
	if t.AlphaTPOT == 0 {
		t.AlphaTPOT = defaultAlphaD
	}
	if t.LogisticScale == 0 {
		t.LogisticScale = defaultScale
	}
	if t.DefaultEpsilonTTFT < 0 || t.DefaultEpsilonTTFT > 1 {
		return fmt.Errorf("defaultEpsilonTtft must be in [0,1], got %f", t.DefaultEpsilonTTFT)
	}
	if t.DefaultEpsilonTPOT < 0 || t.DefaultEpsilonTPOT > 1 {
		return fmt.Errorf("defaultEpsilonTpot must be in [0,1], got %f", t.DefaultEpsilonTPOT)
	}
	if t.AlphaTTFT < 0 || t.AlphaTPOT < 0 {
		return fmt.Errorf("alphaTtft and alphaTpot must be >= 0")
	}
	if t.LogisticScale <= 0 {
		return fmt.Errorf("logisticScale must be > 0, got %f", t.LogisticScale)
	}
	seen := make(map[string]struct{}, len(t.Tiers))
	for i := range t.Tiers {
		tier := &t.Tiers[i]
		if tier.Name == "" {
			return fmt.Errorf("tiers[%d]: name is required", i)
		}
		if _, dup := seen[tier.Name]; dup {
			return fmt.Errorf("duplicate tier name %q", tier.Name)
		}
		seen[tier.Name] = struct{}{}
		if tier.TTFTSLOMs < 0 || tier.TPOTSLOMs < 0 {
			return fmt.Errorf("tier %q: SLO thresholds must be >= 0", tier.Name)
		}
		if tier.EpsilonTTFT == 0 {
			tier.EpsilonTTFT = t.DefaultEpsilonTTFT
		}
		if tier.EpsilonTPOT == 0 {
			tier.EpsilonTPOT = t.DefaultEpsilonTPOT
		}
		if tier.EpsilonTTFT < 0 || tier.EpsilonTTFT > 1 || tier.EpsilonTPOT < 0 || tier.EpsilonTPOT > 1 {
			return fmt.Errorf("tier %q: epsilon must be in [0,1]", tier.Name)
		}
		if tier.Weight < 0 {
			return fmt.Errorf("tier %q: weight must be >= 0", tier.Name)
		}
		if tier.Weight == 0 {
			tier.Weight = 1
		}
	}
	return nil
}

// ResolveRequestSLO maps request headers onto a RequestSLO using the tier table.
// Precedence: named tier header → per-request SLO headers → first configured tier.
func (t TierTable) ResolveRequestSLO(headers map[string]string) (RequestSLO, error) {
	out := RequestSLO{
		EpsilonTTFT: t.DefaultEpsilonTTFT,
		EpsilonTPOT: t.DefaultEpsilonTPOT,
		AlphaTTFT:   t.AlphaTTFT,
		AlphaTPOT:   t.AlphaTPOT,
		Scale:       t.LogisticScale,
		Weight:      1,
	}
	if out.EpsilonTTFT == 0 {
		out.EpsilonTTFT = defaultEpsilon
	}
	if out.EpsilonTPOT == 0 {
		out.EpsilonTPOT = defaultEpsilon
	}
	if out.AlphaTTFT == 0 {
		out.AlphaTTFT = defaultAlphaP
	}
	if out.AlphaTPOT == 0 {
		out.AlphaTPOT = defaultAlphaD
	}
	if out.Scale == 0 {
		out.Scale = defaultScale
	}

	if name, ok := metadata.GetLowerCaseHeaderValue(headers, TierHeaderKey); ok && name != "" {
		for _, tier := range t.Tiers {
			if tier.Name == name {
				out.TierName = tier.Name
				out.TTFTSLOMs = tier.TTFTSLOMs
				out.TPOTSLOMs = tier.TPOTSLOMs
				out.EpsilonTTFT = tier.EpsilonTTFT
				out.EpsilonTPOT = tier.EpsilonTPOT
				out.Weight = tier.Weight
				return out, nil
			}
		}
		return out, fmt.Errorf("unknown SLA tier %q", name)
	}

	ttft, err := parseSLOHeader(headers, metadata.TTFTSLOHeaderKey)
	if err != nil {
		return out, err
	}
	tpot, err := parseSLOHeader(headers, metadata.TPOTSLOHeaderKey)
	if err != nil {
		return out, err
	}
	if ttft > 0 || tpot > 0 {
		out.TTFTSLOMs = ttft
		out.TPOTSLOMs = tpot
		return out, nil
	}

	if len(t.Tiers) > 0 {
		tier := t.Tiers[0]
		out.TierName = tier.Name
		out.TTFTSLOMs = tier.TTFTSLOMs
		out.TPOTSLOMs = tier.TPOTSLOMs
		out.EpsilonTTFT = tier.EpsilonTTFT
		out.EpsilonTPOT = tier.EpsilonTPOT
		out.Weight = tier.Weight
		return out, nil
	}
	return out, nil
}

func parseSLOHeader(headers map[string]string, key string) (float64, error) {
	raw, ok := metadata.GetLowerCaseHeaderValue(headers, key)
	if !ok || raw == "" {
		return 0, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s header %q: %w", key, raw, err)
	}
	if v < 0 {
		return 0, fmt.Errorf("%s must be >= 0, got %f", key, v)
	}
	return v, nil
}

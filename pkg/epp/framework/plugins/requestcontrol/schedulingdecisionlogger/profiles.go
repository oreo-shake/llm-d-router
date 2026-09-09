/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    10|Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package schedulingdecisionlogger

import (
	"fmt"
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/ttftcost"
)

// Component types for counterfactualProfiles[].scorers[].type.
const (
	componentPrefix = "prefix"  // match_blocks / n (prefix-cache-scorer default)
	componentTTFT   = "ttft"    // 1 - cost/cost_cold (ttft-cost-scorer normalize)
	componentQueue  = "queue"   // queue-scorer min-max inverse depth
	componentKVUtil = "kv-util" // 1 - kv_cache_usage_percent
)

// ScorerRef is one weighted component inside a counterfactual profile.
type ScorerRef struct {
	// Type is prefix | ttft | queue | kv-util.
	Type string `json:"type"`
	// Weight defaults to 1 when omitted or zero.
	Weight float64 `json:"weight,omitempty"`
}

// CounterfactualProfileConfig names a same-tick weighted sum of components.
type CounterfactualProfileConfig struct {
	Name    string      `json:"name"`
	Scorers []ScorerRef `json:"scorers"`
}

type compiledProfile struct {
	name    string
	scorers []compiledScorer
}

type compiledScorer struct {
	typ    string
	weight float64
}

func compileProfiles(cfgs []CounterfactualProfileConfig, iib *ttftcost.Weights) ([]compiledProfile, error) {
	if len(cfgs) == 0 {
		return nil, nil
	}
	seen := make(map[string]struct{}, len(cfgs))
	out := make([]compiledProfile, 0, len(cfgs))
	for _, cfg := range cfgs {
		if cfg.Name == "" {
			return nil, fmt.Errorf("counterfactualProfiles: name is required")
		}
		if _, ok := seen[cfg.Name]; ok {
			return nil, fmt.Errorf("counterfactualProfiles: duplicate name %q", cfg.Name)
		}
		seen[cfg.Name] = struct{}{}
		if len(cfg.Scorers) == 0 {
			return nil, fmt.Errorf("counterfactualProfiles %q: scorers is required", cfg.Name)
		}
		comps := make([]compiledScorer, 0, len(cfg.Scorers))
		for _, s := range cfg.Scorers {
			typ := s.Type
			switch typ {
			case componentPrefix, componentQueue, componentKVUtil:
			case componentTTFT:
				if iib == nil {
					return nil, fmt.Errorf("counterfactualProfiles %q: type %q requires iibWeights", cfg.Name, typ)
				}
			default:
				return nil, fmt.Errorf("counterfactualProfiles %q: unknown scorer type %q (want %s|%s|%s|%s)",
					cfg.Name, typ, componentPrefix, componentTTFT, componentQueue, componentKVUtil)
			}
			w := s.Weight
			if w == 0 {
				w = 1
			}
			if w < 0 || math.IsNaN(w) {
				return nil, fmt.Errorf("counterfactualProfiles %q: weight must be >= 0", cfg.Name)
			}
			comps = append(comps, compiledScorer{typ: typ, weight: w})
		}
		out = append(out, compiledProfile{name: cfg.Name, scorers: comps})
	}
	return out, nil
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func prefixComponent(c CandidateFeatures) float64 {
	if c.N <= 0 {
		return 0
	}
	return clamp01(c.MatchBlocks / float64(c.N))
}

func ttftComponent(c CandidateFeatures, w ttftcost.Weights) float64 {
	if c.N <= 0 {
		return 0
	}
	cost := ttftcost.Cost(w, c.N, c.MGPU, c.MCPU)
	cold := ttftcost.Cost(w, c.N, 0, 0)
	if cold <= 0 {
		return clamp01(-cost)
	}
	return clamp01(1.0 - cost/cold)
}

func queueComponents(cands []CandidateFeatures) []float64 {
	out := make([]float64, len(cands))
	if len(cands) == 0 {
		return out
	}
	minQ, maxQ := cands[0].WaitingQueueSize, cands[0].WaitingQueueSize
	for i := 1; i < len(cands); i++ {
		q := cands[i].WaitingQueueSize
		if q < minQ {
			minQ = q
		}
		if q > maxQ {
			maxQ = q
		}
	}
	if maxQ == minQ {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	den := float64(maxQ - minQ)
	for i, c := range cands {
		out[i] = float64(maxQ-c.WaitingQueueSize) / den
	}
	return out
}

func kvUtilComponent(c CandidateFeatures) float64 {
	return clamp01(1.0 - c.KVCacheUsagePercent)
}

// pickProfile returns argmax of the weighted component sum. Ties break by
// endpoint id ascending. Score is the combined weighted sum (may exceed 1).
func pickProfile(cands []CandidateFeatures, profile compiledProfile, iib *ttftcost.Weights) *CounterfactualPick {
	if len(cands) == 0 {
		return nil
	}
	queueScores := queueComponents(cands)
	bestIdx := 0
	bestScore := profileScore(cands[0], 0, profile, iib, queueScores)
	for i := 1; i < len(cands); i++ {
		s := profileScore(cands[i], i, profile, iib, queueScores)
		if s > bestScore || (s == bestScore && cands[i].Endpoint < cands[bestIdx].Endpoint) {
			bestIdx = i
			bestScore = s
		}
	}
	return &CounterfactualPick{Endpoint: cands[bestIdx].Endpoint, Score: bestScore}
}

func profileScore(c CandidateFeatures, idx int, profile compiledProfile, iib *ttftcost.Weights, queueScores []float64) float64 {
	var sum float64
	for _, s := range profile.scorers {
		var part float64
		switch s.typ {
		case componentPrefix:
			part = prefixComponent(c)
		case componentTTFT:
			part = ttftComponent(c, *iib)
		case componentQueue:
			part = queueScores[idx]
		case componentKVUtil:
			part = kvUtilComponent(c)
		}
		sum += s.weight * part
	}
	return sum
}

// profilesDisagree reports whether any two profile picks select different pods.
func profilesDisagree(profiles map[string]*CounterfactualPick) bool {
	var first string
	have := false
	for _, p := range profiles {
		if p == nil {
			continue
		}
		if !have {
			first = p.Endpoint
			have = true
			continue
		}
		if p.Endpoint != first {
			return true
		}
	}
	return false
}

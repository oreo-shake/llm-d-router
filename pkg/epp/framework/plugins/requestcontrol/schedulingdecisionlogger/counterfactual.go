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

package schedulingdecisionlogger

import "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/ttftcost"

// pickMxW returns argmax(match_blocks). Ties break by endpoint id ascending.
func pickMxW(cands []CandidateFeatures) *CounterfactualPick {
	if len(cands) == 0 {
		return nil
	}
	best := cands[0]
	for i := 1; i < len(cands); i++ {
		c := cands[i]
		if c.MatchBlocks > best.MatchBlocks ||
			(c.MatchBlocks == best.MatchBlocks && c.Endpoint < best.Endpoint) {
			best = c
		}
	}
	return &CounterfactualPick{Endpoint: best.Endpoint, Score: best.MatchBlocks}
}

// pickIIB returns argmin(Method IIB cost). Ties break by endpoint id ascending.
func pickIIB(cands []CandidateFeatures, w ttftcost.Weights) *CounterfactualPick {
	if len(cands) == 0 {
		return nil
	}
	best := cands[0]
	bestCost := ttftcost.Cost(w, best.N, best.MGPU, best.MCPU)
	for i := 1; i < len(cands); i++ {
		c := cands[i]
		cost := ttftcost.Cost(w, c.N, c.MGPU, c.MCPU)
		if cost < bestCost || (cost == bestCost && c.Endpoint < best.Endpoint) {
			best = c
			bestCost = cost
		}
	}
	return &CounterfactualPick{Endpoint: best.Endpoint, Cost: bestCost}
}

// isGateCase is the Method IIB methodology pattern: m×w prefers a predominantly
// GPU match with an unmatched tail, while IIB prefers a fuller CPU match.
func isGateCase(cands []CandidateFeatures, mxw, iib *CounterfactualPick) bool {
	if mxw == nil || iib == nil || mxw.Endpoint == iib.Endpoint {
		return false
	}
	var mxwC, iibC *CandidateFeatures
	for i := range cands {
		c := &cands[i]
		if c.Endpoint == mxw.Endpoint {
			mxwC = c
		}
		if c.Endpoint == iib.Endpoint {
			iibC = c
		}
	}
	if mxwC == nil || iibC == nil {
		return false
	}
	mxwUnmatched := mxwC.N - mxwC.M
	iibUnmatched := iibC.N - iibC.M
	gpuDominant := mxwC.MGPU > 0 && mxwC.MGPU >= mxwC.MCPU && mxwUnmatched > 0
	cpuFuller := iibC.MCPU > mxwC.MCPU && iibUnmatched < mxwUnmatched
	return gpuDominant && cpuFuller
}

func boolPtr(v bool) *bool { return &v }

// agree reports whether chosen matches a counterfactual endpoint.
func agree(chosen string, pick *CounterfactualPick) *bool {
	if pick == nil || chosen == "" {
		return nil
	}
	return boolPtr(chosen == pick.Endpoint)
}

// disagreesWithAny is true when at least one configured counterfactual differs
// from the live chosen endpoint.
func disagreesWithAny(chosen string, cf Counterfactuals) bool {
	if cf.MxW != nil && cf.MxW.Endpoint != chosen {
		return true
	}
	if cf.IIB != nil && cf.IIB.Endpoint != chosen {
		return true
	}
	for _, p := range cf.Profiles {
		if p != nil && p.Endpoint != chosen {
			return true
		}
	}
	return false
}

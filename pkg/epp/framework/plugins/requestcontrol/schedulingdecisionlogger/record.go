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

import "time"

// CandidateFeatures is the per-endpoint snapshot used for ranking comparison
// across experiments (prefix match plus optional live queue/KV metrics).
type CandidateFeatures struct {
	Endpoint            string  `json:"endpoint"`
	N                   int     `json:"n"`
	M                   int     `json:"m"`
	MGPU                int     `json:"m_gpu"`
	MCPU                int     `json:"m_cpu"`
	MatchBlocks         float64 `json:"match_blocks"`
	WaitingQueueSize    int     `json:"waiting_queue_size,omitempty"`
	KVCacheUsagePercent float64 `json:"kv_cache_usage_percent,omitempty"`
	// SLA-aware routing fields (populated when latency / SLARisk attributes exist).
	PredictedTTFTMs *float64 `json:"predicted_ttft_ms,omitempty"`
	PredictedTPOTMs *float64 `json:"predicted_tpot_ms,omitempty"`
	TTFTSurvival    *float64 `json:"ttft_survival,omitempty"`
	TPOTSurvival    *float64 `json:"tpot_survival,omitempty"`
	SLAFeasible     *bool    `json:"sla_feasible,omitempty"`
	SLAFutureCost   *float64 `json:"sla_future_cost,omitempty"`
	SLARiskSource   string   `json:"sla_risk_source,omitempty"`
}

// CounterfactualPick is one policy's argmax/argmin over the same candidate set.
type CounterfactualPick struct {
	Endpoint string  `json:"endpoint"`
	Score    float64 `json:"score,omitempty"`
	Cost     float64 `json:"cost,omitempty"`
}

// Counterfactuals holds same-tick alternative picks. Fields are omitted when
// the corresponding policy could not be evaluated (missing features / weights).
type Counterfactuals struct {
	MxW *CounterfactualPick `json:"mxw,omitempty"`
	IIB *CounterfactualPick `json:"iib,omitempty"`
	// Profiles maps counterfactualProfiles[].name to that profile's pick.
	Profiles map[string]*CounterfactualPick `json:"profiles,omitempty"`
}

// DecisionRecord is one schedule decision. Schema is stable across experiments
// so JSONL files can be compared without request-id joins.
type DecisionRecord struct {
	Experiment     string              `json:"experiment"`
	TS             time.Time           `json:"ts"`
	RequestID      string              `json:"request_id,omitempty"`
	Profile        string              `json:"profile"`
	Chosen         string              `json:"chosen"`
	ScorerScores   map[string]float64  `json:"scorer_scores,omitempty"`
	Candidates     []CandidateFeatures `json:"candidates"`
	Counterfactual Counterfactuals     `json:"counterfactual"`
	AgreeMxW       *bool               `json:"agree_mxw,omitempty"`
	AgreeIIB       *bool               `json:"agree_iib,omitempty"`
	// AgreeProfiles maps profile name -> whether chosen matches that pick.
	AgreeProfiles map[string]*bool `json:"agree_profiles,omitempty"`
	// ProfilesDisagree is true when any two configured CF profiles pick different pods.
	ProfilesDisagree bool `json:"profiles_disagree,omitempty"`
	GateCase         bool `json:"gate_case"`
}

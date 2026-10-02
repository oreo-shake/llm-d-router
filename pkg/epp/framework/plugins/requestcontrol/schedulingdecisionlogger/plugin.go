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

// Package schedulingdecisionlogger records per-request scheduling decisions
// for cross-experiment comparison (chosen pod, candidate features, and
// same-tick m×w / Method IIB / configurable multi-scorer counterfactuals).
package schedulingdecisionlogger

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrslarisk "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/slarisk"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/ttftcost"
)

const (
	// PluginType is the registered type name.
	PluginType = "scheduling-decision-logger"

	defaultSampleRate = 1.0
)

// Config is the JSON parameters for scheduling-decision-logger.
type Config struct {
	// Experiment labels metrics and records (e.g. "iib-offline", "prefix-scalar-0.8").
	Experiment string `json:"experiment"`
	// PrefixMatchInfoProducerName selects which PrefixCacheMatchInfo producer
	// to read. Required.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
	// LatencyPredictionInfoProducerName selects LatencyPredictionInfo (optional SLA audit).
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	// SLARiskInfoProducerName selects SLARiskInfo (optional SLA audit).
	SLARiskInfoProducerName string `json:"slaRiskInfoProducerName,omitempty"`
	// OutputPath appends JSONL decision records. Empty emits structured logs only.
	OutputPath string `json:"outputPath,omitempty"`
	// IIBWeights enables counterfactual.iib and any profile component type
	// "ttft" via Method IIB Cost. Omit to skip those.
	IIBWeights *ttftcost.Weights `json:"iibWeights,omitempty"`
	// CounterfactualProfiles are named same-tick weighted sums of component
	// scorers (prefix|ttft|queue|kv-util). Use to compare combos such as
	// prefix+queue+kv vs ttft+queue+kv without changing live routing.
	CounterfactualProfiles []CounterfactualProfileConfig `json:"counterfactualProfiles,omitempty"`
	// SampleRate is the probability of emitting an agree record when LogAgrees
	// is true. Disagreements are always emitted. Default 1.0.
	SampleRate float64 `json:"sampleRate,omitempty"`
	// LogAgrees emits records where chosen matches every configured
	// counterfactual. Default false (disagreements only, plus metrics always).
	LogAgrees bool `json:"logAgrees,omitempty"`
}

// Plugin implements requestcontrol.PreRequest.
type Plugin struct {
	typedName                    plugin.TypedName
	prefixMatchDataKey           plugin.DataKey
	latencyPredictionInfoDataKey plugin.DataKey
	slaRiskDataKey               plugin.DataKey
	experiment                   string
	sampleRate                   float64
	logAgrees                    bool
	iibWeights                   *ttftcost.Weights
	profiles                     []compiledProfile
	writer                       *jsonlWriter
}

var _ requestcontrol.PreRequest = &Plugin{}

// Factory constructs a scheduling-decision-logger from JSON parameters.
func Factory(name string, decoder *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{SampleRate: defaultSampleRate}
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to decode %s parameters: %w", PluginType, err)
		}
	}
	return New(name, cfg, handle)
}

// New builds a Plugin. Used by Factory and tests.
func New(name string, cfg Config, handle plugin.Handle) (*Plugin, error) {
	if cfg.Experiment == "" {
		return nil, fmt.Errorf("%s: experiment is required", PluginType)
	}
	if cfg.PrefixMatchInfoProducerName == "" {
		return nil, fmt.Errorf("%s: prefixMatchInfoProducerName is required", PluginType)
	}
	sampleRate := cfg.SampleRate
	if sampleRate == 0 {
		sampleRate = defaultSampleRate
	}
	if sampleRate < 0 || sampleRate > 1 || math.IsNaN(sampleRate) {
		return nil, fmt.Errorf("%s: sampleRate must be in [0,1], got %v", PluginType, cfg.SampleRate)
	}
	if cfg.IIBWeights != nil {
		w := *cfg.IIBWeights
		if w.CPU < 0 || w.UnmatchedLinear < 0 || w.UnmatchedQuadratic < 0 || w.GPU < 0 {
			return nil, fmt.Errorf("%s: iibWeights must be >= 0", PluginType)
		}
	}
	profiles, err := compileProfiles(cfg.CounterfactualProfiles, cfg.IIBWeights)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", PluginType, err)
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}

	p := &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		prefixMatchDataKey: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(
			cfg.PrefixMatchInfoProducerName,
		),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(
			cfg.LatencyPredictionInfoProducerName,
		),
		slaRiskDataKey: attrslarisk.SLARiskInfoDataKey.WithNonEmptyProducerName(
			cfg.SLARiskInfoProducerName,
		),
		experiment: cfg.Experiment,
		sampleRate: sampleRate,
		logAgrees:  cfg.LogAgrees,
		iibWeights: cfg.IIBWeights,
		profiles:   profiles,
	}
	if cfg.OutputPath != "" {
		w, err := openJSONL(cfg.OutputPath)
		if err != nil {
			return nil, err
		}
		p.writer = w
	}
	return p, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

// PreRequest records the primary-profile scheduling decision. Audit failures
// never fail the request.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	logger := log.FromContext(ctx)
	rec, err := p.buildRecord(request, schedulingResult)
	if err != nil {
		logger.V(logutil.DEBUG).Info("scheduling decision logger skipped", "reason", err.Error())
		return nil
	}
	recordMetrics(p.experiment, rec.AgreeMxW, rec.GateCase)

	if !p.shouldEmit(rec) {
		return nil
	}
	if p.writer != nil {
		if werr := p.writer.Write(rec); werr != nil {
			logger.Error(werr, "failed to write scheduling decision record")
		}
	}
	// Always Info (not V(DEFAULT)): EPP often runs with --v=1, which filters
	// V(2). When outputPath is set, still Info so kubectl logs remain usable.
	logger.Info("scheduling decision", "decision", rec)
	return nil
}

func (p *Plugin) shouldEmit(rec *DecisionRecord) bool {
	if disagreesWithAny(rec.Chosen, rec.Counterfactual) {
		return true
	}
	if !p.logAgrees {
		return false
	}
	if p.sampleRate >= 1 {
		return true
	}
	return rand.Float64() < p.sampleRate
}

func (p *Plugin) buildRecord(request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) (*DecisionRecord, error) {
	if schedulingResult == nil {
		return nil, fmt.Errorf("nil scheduling result")
	}
	profile := schedulingResult.PrimaryProfileName
	pr := schedulingResult.ProfileResults[profile]
	if pr == nil {
		return nil, fmt.Errorf("missing primary profile %q", profile)
	}

	chosen := ""
	if len(pr.TargetEndpoints) > 0 && pr.TargetEndpoints[0] != nil && pr.TargetEndpoints[0].GetMetadata() != nil {
		chosen = pr.TargetEndpoints[0].GetMetadata().ID.String()
	}

	endpoints := endpointsFromProfile(pr)
	cands := make([]CandidateFeatures, 0, len(endpoints))
	scores := make(map[string]float64, len(pr.ScoredCandidates))
	for _, se := range pr.ScoredCandidates {
		if se.Endpoint == nil || se.Endpoint.GetMetadata() == nil {
			continue
		}
		scores[se.Endpoint.GetMetadata().ID.String()] = se.Score
	}

	missingInfo := 0
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		id := ep.GetMetadata().ID.String()
		raw, ok := ep.Get(p.prefixMatchDataKey)
		if !ok {
			missingInfo++
			continue
		}
		info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
		if !ok {
			missingInfo++
			continue
		}
		n := info.TotalBlocks()
		m := info.CachedBlockCount()
		if m < 0 {
			m = 0
		}
		if n > 0 && m > n {
			m = n
		}
		mGPU, mCPU := ttftcost.Attribution(m, info.CachedBlocksByTier())
		cand := CandidateFeatures{
			Endpoint:    id,
			N:           n,
			M:           m,
			MGPU:        mGPU,
			MCPU:        mCPU,
			MatchBlocks: info.MatchBlocks(),
		}
		if metrics := ep.GetMetrics(); metrics != nil {
			cand.WaitingQueueSize = metrics.WaitingQueueSize
			cand.KVCacheUsagePercent = metrics.KVCacheUsagePercent
		}
		enrichSLACandidate(&cand, ep, p.slaRiskDataKey, p.latencyPredictionInfoDataKey)
		cands = append(cands, cand)
	}
	if len(cands) == 0 {
		return nil, fmt.Errorf("no candidates with PrefixCacheMatchInfo (missing=%d)", missingInfo)
	}

	cf := Counterfactuals{MxW: pickMxW(cands)}
	if p.iibWeights != nil {
		cf.IIB = pickIIB(cands, *p.iibWeights)
	}
	var agreeProfiles map[string]*bool
	if len(p.profiles) > 0 {
		cf.Profiles = make(map[string]*CounterfactualPick, len(p.profiles))
		agreeProfiles = make(map[string]*bool, len(p.profiles))
		for _, prof := range p.profiles {
			pick := pickProfile(cands, prof, p.iibWeights)
			cf.Profiles[prof.name] = pick
			agreeProfiles[prof.name] = agree(chosen, pick)
		}
	}

	reqID := ""
	if request != nil {
		reqID = request.RequestID
	}
	return &DecisionRecord{
		Experiment:       p.experiment,
		TS:               time.Now().UTC(),
		RequestID:        reqID,
		Profile:          profile,
		Chosen:           chosen,
		ScorerScores:     scores,
		Candidates:       cands,
		Counterfactual:   cf,
		AgreeMxW:         agree(chosen, cf.MxW),
		AgreeIIB:         agree(chosen, cf.IIB),
		AgreeProfiles:    agreeProfiles,
		ProfilesDisagree: profilesDisagree(cf.Profiles),
		GateCase:         isGateCase(cands, cf.MxW, cf.IIB),
	}, nil
}

// endpointsFromProfile prefers ScoredCandidates (full scored set); falls back
// to TargetEndpoints when scorers did not populate ScoredCandidates.
func endpointsFromProfile(pr *fwksched.ProfileRunResult) []fwksched.Endpoint {
	if len(pr.ScoredCandidates) > 0 {
		out := make([]fwksched.Endpoint, 0, len(pr.ScoredCandidates))
		for _, se := range pr.ScoredCandidates {
			out = append(out, se.Endpoint)
		}
		return out
	}
	return pr.TargetEndpoints
}

func enrichSLACandidate(cand *CandidateFeatures, ep fwksched.Endpoint, riskKey, latencyKey plugin.DataKey) {
	if raw, ok := ep.Get(riskKey); ok {
		if info, ok := raw.(*attrslarisk.SLARiskInfo); ok {
			ttft, tpot := info.TTFTPred(), info.TPOTPred()
			ttftS, tpotS := info.TTFTSurvival(), info.TPOTSurvival()
			feas := info.Feasible()
			cand.PredictedTTFTMs = &ttft
			cand.PredictedTPOTMs = &tpot
			cand.TTFTSurvival = &ttftS
			cand.TPOTSurvival = &tpotS
			cand.SLAFeasible = &feas
			cand.SLARiskSource = info.Source()
			if info.HasFuture() {
				v := info.FutureCost()
				cand.SLAFutureCost = &v
			}
			return
		}
	}
	if raw, ok := ep.Get(latencyKey); ok {
		if info, ok := raw.(*attrlatency.LatencyPredictionInfo); ok {
			ttft, tpot := info.TTFT(), info.TPOT()
			cand.PredictedTTFTMs = &ttft
			cand.PredictedTPOTMs = &tpot
		}
	}
}

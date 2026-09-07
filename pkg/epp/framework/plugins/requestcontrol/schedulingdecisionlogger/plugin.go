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
// same-tick m×w / Method IIB counterfactuals).
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
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
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
	// OutputPath appends JSONL decision records. Empty emits structured logs only.
	OutputPath string `json:"outputPath,omitempty"`
	// IIBWeights enables counterfactual.iib via Method IIB Cost. Omit to skip.
	IIBWeights *ttftcost.Weights `json:"iibWeights,omitempty"`
	// SampleRate is the probability of emitting an agree record when LogAgrees
	// is true. Disagreements are always emitted. Default 1.0.
	SampleRate float64 `json:"sampleRate,omitempty"`
	// LogAgrees emits records where chosen matches every configured
	// counterfactual. Default false (disagreements only, plus metrics always).
	LogAgrees bool `json:"logAgrees,omitempty"`
}

// Plugin implements requestcontrol.PreRequest.
type Plugin struct {
	typedName          plugin.TypedName
	prefixMatchDataKey plugin.DataKey
	experiment         string
	sampleRate         float64
	logAgrees          bool
	iibWeights         *ttftcost.Weights
	writer             *jsonlWriter
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
		experiment: cfg.Experiment,
		sampleRate: sampleRate,
		logAgrees:  cfg.LogAgrees,
		iibWeights: cfg.IIBWeights,
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
		cands = append(cands, CandidateFeatures{
			Endpoint:    id,
			N:           n,
			M:           m,
			MGPU:        mGPU,
			MCPU:        mCPU,
			MatchBlocks: info.MatchBlocks(),
		})
	}
	if len(cands) == 0 {
		return nil, fmt.Errorf("no candidates with PrefixCacheMatchInfo (missing=%d)", missingInfo)
	}

	cf := Counterfactuals{MxW: pickMxW(cands)}
	if p.iibWeights != nil {
		cf.IIB = pickIIB(cands, *p.iibWeights)
	}

	reqID := ""
	if request != nil {
		reqID = request.RequestID
	}
	return &DecisionRecord{
		Experiment:     p.experiment,
		TS:             time.Now().UTC(),
		RequestID:      reqID,
		Profile:        profile,
		Chosen:         chosen,
		ScorerScores:   scores,
		Candidates:     cands,
		Counterfactual: cf,
		AgreeMxW:       agree(chosen, cf.MxW),
		AgreeIIB:       agree(chosen, cf.IIB),
		GateCase:       isGateCase(cands, cf.MxW, cf.IIB),
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

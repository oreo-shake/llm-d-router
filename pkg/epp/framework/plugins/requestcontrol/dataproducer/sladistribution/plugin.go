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

// Package sladistribution produces per-endpoint SLARiskInfo from latency
// predictions. Optionally calls the BLIS placement-eval sidecar for V_H
// (tier-weighted future SLA violation counts) and attaches it as FutureCost.
package sladistribution

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrslarisk "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/slarisk"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/placementeval/placementevalclient"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sladistribution/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/slaaware"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type name.
	PluginType = constants.PluginType

	modeQuantile = "quantile"
	modeCDF      = "cdf"

	futureModeHistory   = "recent-history"
	futureModeSynthetic = "synthetic"
)

var _ requestcontrol.DataProducer = &Plugin{}

// Config is the JSON parameters for sla-distribution-producer.
type Config struct {
	slaaware.TierTable
	// Mode is "quantile" (default) or "cdf".
	Mode string `json:"mode,omitempty"`

	// EnablePlacementEval calls the BLIS sidecar for V_H and sets FutureCost.
	EnablePlacementEval bool `json:"enablePlacementEval,omitempty"`
	// PlacementEvalURL overrides PLACEMENT_EVAL_URL (default http://localhost:9091).
	PlacementEvalURL string `json:"placementEvalUrl,omitempty"`
	// PlacementEvalTimeoutMs overrides PLACEMENT_EVAL_TIMEOUT_MS (default 50).
	PlacementEvalTimeoutMs int `json:"placementEvalTimeoutMs,omitempty"`
	// HorizonUs is the simulation horizon for V_H (default 2e6 = 2s).
	HorizonUs int64 `json:"horizonUs,omitempty"`
	// FutureWorkloadMode is "recent-history" (default) or "synthetic".
	FutureWorkloadMode string `json:"futureWorkloadMode,omitempty"`
	// HistoryMaxRequests / HistoryMaxAgeSec size the recent-history ring buffer.
	HistoryMaxRequests int `json:"historyMaxRequests,omitempty"`
	HistoryMaxAgeSec   int `json:"historyMaxAgeSec,omitempty"`
	// SyntheticRatePerSec / SyntheticPromptTokens / SyntheticOutputTokens / SyntheticSLOClass
	// configure synthetic future windows.
	SyntheticRatePerSec    float64 `json:"syntheticRatePerSec,omitempty"`
	SyntheticPromptTokens  int     `json:"syntheticPromptTokens,omitempty"`
	SyntheticOutputTokens  int     `json:"syntheticOutputTokens,omitempty"`
	SyntheticSLOClass      string  `json:"syntheticSloClass,omitempty"`
	// Hydrate template for sidecar pods (required fields for BLIS HydrateInstance).
	MaxNumSeqs          int `json:"maxNumSeqs,omitempty"`
	MaxNumBatchedTokens int `json:"maxNumBatchedTokens,omitempty"`
	TotalKVBlocks       int `json:"totalKVBlocks,omitempty"`
	BlockSizeTokens     int `json:"blockSizeTokens,omitempty"`

	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
}

// DefaultConfig is used when parameters are omitted.
var DefaultConfig = Config{
	Mode:                modeQuantile,
	HorizonUs:           2_000_000,
	FutureWorkloadMode:  futureModeHistory,
	HistoryMaxRequests:  256,
	HistoryMaxAgeSec:    30,
	SyntheticRatePerSec: 10,
	SyntheticPromptTokens: 128,
	SyntheticOutputTokens: 32,
	MaxNumSeqs:          8,
	MaxNumBatchedTokens: 4096,
	TotalKVBlocks:       128,
	BlockSizeTokens:     16,
}

// Plugin implements DataProducer for SLA risk attributes.
type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	latencyPredictionInfoDataKey fwkplugin.DataKey
	slaRiskDataKey               fwkplugin.DataKey
	history                      *slaaware.HistoryBuffer
	evalClient                   *placementevalclient.Client
}

// Factory constructs the producer.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal %s config: %w", PluginType, err)
		}
	}
	if err := config.Normalize(); err != nil {
		return nil, err
	}
	if config.Mode == "" {
		config.Mode = modeQuantile
	}
	if config.Mode != modeQuantile && config.Mode != modeCDF {
		return nil, fmt.Errorf("mode must be %q or %q, got %q", modeQuantile, modeCDF, config.Mode)
	}
	if config.HorizonUs == 0 {
		config.HorizonUs = DefaultConfig.HorizonUs
	}
	if config.FutureWorkloadMode == "" {
		config.FutureWorkloadMode = futureModeHistory
	}
	if config.FutureWorkloadMode != futureModeHistory && config.FutureWorkloadMode != futureModeSynthetic {
		return nil, fmt.Errorf("futureWorkloadMode must be %q or %q", futureModeHistory, futureModeSynthetic)
	}
	applyHydrateDefaults(&config)

	p := &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		slaRiskDataKey:               attrslarisk.SLARiskInfoDataKey.WithNonEmptyProducerName(name),
		history:                      slaaware.NewHistoryBuffer(config.HistoryMaxRequests, time.Duration(config.HistoryMaxAgeSec)*time.Second),
	}
	if config.EnablePlacementEval {
		cfg := placementevalclient.ConfigFromEnv()
		if config.PlacementEvalURL != "" {
			cfg.URL = config.PlacementEvalURL
		}
		if config.PlacementEvalTimeoutMs > 0 {
			cfg.Timeout = time.Duration(config.PlacementEvalTimeoutMs) * time.Millisecond
		}
		p.evalClient = placementevalclient.New(cfg)
	}
	return p, nil
}

func applyHydrateDefaults(c *Config) {
	if c.MaxNumSeqs == 0 {
		c.MaxNumSeqs = DefaultConfig.MaxNumSeqs
	}
	if c.MaxNumBatchedTokens == 0 {
		c.MaxNumBatchedTokens = DefaultConfig.MaxNumBatchedTokens
	}
	if c.TotalKVBlocks == 0 {
		c.TotalKVBlocks = DefaultConfig.TotalKVBlocks
	}
	if c.BlockSizeTokens == 0 {
		c.BlockSizeTokens = DefaultConfig.BlockSizeTokens
	}
	if c.HistoryMaxRequests == 0 {
		c.HistoryMaxRequests = DefaultConfig.HistoryMaxRequests
	}
	if c.HistoryMaxAgeSec == 0 {
		c.HistoryMaxAgeSec = DefaultConfig.HistoryMaxAgeSec
	}
	if c.SyntheticRatePerSec == 0 {
		c.SyntheticRatePerSec = DefaultConfig.SyntheticRatePerSec
	}
	if c.SyntheticPromptTokens == 0 {
		c.SyntheticPromptTokens = DefaultConfig.SyntheticPromptTokens
	}
	if c.SyntheticOutputTokens == 0 {
		c.SyntheticOutputTokens = DefaultConfig.SyntheticOutputTokens
	}
}

// TypedName returns the plugin identity.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Produce writes SLARiskInfo for each endpoint that has a latency prediction.
// When EnablePlacementEval is set, also attaches V_H from the BLIS sidecar.
func (p *Plugin) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	logger := log.FromContext(ctx)
	headers := map[string]string{}
	if request != nil {
		headers = request.Headers
	}
	slo, err := p.config.ResolveRequestSLO(headers)
	if err != nil {
		return err
	}

	p.recordHistory(request, slo)

	attrs := make([]*attrslarisk.SLARiskInfo, len(endpoints))
	for i, ep := range endpoints {
		raw, ok := ep.Get(p.latencyPredictionInfoDataKey)
		if !ok {
			continue
		}
		info := raw.(*attrlatency.LatencyPredictionInfo)
		var risk slaaware.EndpointRisk
		if p.config.Mode == modeCDF {
			ttftSurv := slaaware.SurvivalFromQuantile(info.TTFT(), slo.TTFTSLOMs, slo.Scale)
			tpotSurv := slaaware.SurvivalFromQuantile(info.TPOT(), slo.TPOTSLOMs, slo.Scale)
			risk = slaaware.RiskFromCDF(ttftSurv, tpotSurv, info.TTFT(), info.TPOT(), slo)
		} else {
			risk = slaaware.RiskFromQuantile(info.TTFT(), info.TPOT(), slo)
		}
		attr := attrslarisk.NewSLARiskInfo(
			risk.TTFTPred, risk.TPOTPred,
			risk.TTFTSurvival, risk.TPOTSurvival,
			risk.Feasible, risk.Source,
		)
		attrs[i] = attr
		ep.Put(p.slaRiskDataKey, attr)
		logger.V(logutil.DEBUG).Info("stored SLA risk",
			"pod", ep.GetMetadata().ID.Name,
			"feasible", risk.Feasible,
			"source", risk.Source,
			"ttftSurvival", risk.TTFTSurvival,
			"tpotSurvival", risk.TPOTSurvival)
	}

	if p.evalClient != nil && p.config.EnablePlacementEval {
		p.attachFutureCosts(ctx, request, slo, endpoints, attrs)
	}
	return nil
}

func (p *Plugin) recordHistory(request *fwksched.InferenceRequest, slo slaaware.RequestSLO) {
	if p.history == nil || request == nil {
		return
	}
	inTok, outTok := estimateTokens(request)
	cls := slo.TierName
	if cls == "" {
		cls = "default"
	}
	p.history.Add(slaaware.ArrivalRecord{
		ArrivedAt:    time.Now(),
		InputTokens:  inTok,
		OutputTokens: outTok,
		SLOClass:     cls,
	})
}

func estimateTokens(request *fwksched.InferenceRequest) (in, out int) {
	in, out = 128, 32
	if request == nil || request.Body == nil {
		return in, out
	}
	// Best-effort: OpenAI max_tokens if present via raw body is not typed here;
	// keep defaults unless headers carry hints.
	if v, ok := metadata.GetLowerCaseHeaderValue(request.Headers, "x-llm-d-estimated-input-tokens"); ok {
		if n, err := parsePositiveInt(v); err == nil {
			in = n
		}
	}
	if v, ok := metadata.GetLowerCaseHeaderValue(request.Headers, "x-llm-d-estimated-output-tokens"); ok {
		if n, err := parsePositiveInt(v); err == nil {
			out = n
		}
	}
	return in, out
}

func parsePositiveInt(s string) (int, error) {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("not an int")
		}
		n = n*10 + int(c-'0')
	}
	if n <= 0 {
		return 0, fmt.Errorf("non-positive")
	}
	return n, nil
}

func (p *Plugin) attachFutureCosts(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	slo slaaware.RequestSLO,
	endpoints []fwksched.Endpoint,
	attrs []*attrslarisk.SLARiskInfo,
) {
	logger := log.FromContext(ctx)
	future := p.buildFutureWindow()
	if len(future) == 0 && p.config.FutureWorkloadMode == futureModeHistory {
		// Fall back to synthetic when history is empty.
		future = slaaware.BuildSyntheticWindow(
			p.config.HorizonUs, p.config.SyntheticRatePerSec,
			p.config.SyntheticPromptTokens, p.config.SyntheticOutputTokens,
			p.config.SyntheticSLOClass, "syn",
		)
	}
	if len(future) == 0 {
		logger.V(logutil.DEBUG).Info("placement-eval: empty future window, skip V_H")
		return
	}

	inTok, outTok := estimateTokens(request)
	candID := "cand"
	if request != nil && request.RequestID != "" {
		candID = request.RequestID
	}
	cls := slo.TierName
	if cls == "" {
		cls = "default"
	}

	pods := make([]placementevalclient.PodSpec, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		qd, batch, kv := 0, 0, 0.0
		if m := ep.GetMetrics(); m != nil {
			qd = m.WaitingQueueSize
			batch = m.RunningRequestsSize
			kv = m.KVCacheUsagePercent
		}
		pods = append(pods, placementevalclient.PodSpec{
			PodID:               ep.GetMetadata().ID.String(),
			ClockUs:             0,
			QueueDepth:          qd,
			BatchSize:           batch,
			KVUtilization:       kv,
			MaxNumSeqs:          p.config.MaxNumSeqs,
			MaxNumBatchedTokens: p.config.MaxNumBatchedTokens,
			TotalKVBlocks:       p.config.TotalKVBlocks,
			BlockSizeTokens:     p.config.BlockSizeTokens,
		})
	}
	if len(pods) == 0 {
		return
	}

	sloTargets := map[string]placementevalclient.SLODimTargets{}
	for _, tier := range p.config.Tiers {
		sloTargets[tier.Name] = placementevalclient.SLODimTargets{TTFTMs: tier.TTFTSLOMs, ITLMs: tier.TPOTSLOMs}
	}
	if len(sloTargets) == 0 {
		sloTargets["default"] = placementevalclient.SLODimTargets{TTFTMs: slo.TTFTSLOMs, ITLMs: slo.TPOTSLOMs}
	}

	tierWeights := map[string]float64{}
	for _, tier := range p.config.Tiers {
		tierWeights[tier.Name] = tier.Weight
		if tierWeights[tier.Name] == 0 {
			tierWeights[tier.Name] = 1
		}
	}

	futSpecs := make([]placementevalclient.RequestSpec, 0, len(future))
	for _, f := range future {
		futSpecs = append(futSpecs, placementevalclient.RequestSpec{
			ID: f.ID, ArrivalTime: f.ArrivalTimeUs,
			InputTokens: f.InputTokens, OutputTokens: f.OutputTokens, SLOClass: f.SLOClass,
		})
	}

	evalReq := placementevalclient.EvaluateRequest{
		HorizonUs:  p.config.HorizonUs,
		SLOTargets: sloTargets,
		CostWeights: placementevalclient.CostWeights{
			AlphaPrefill: p.config.AlphaTTFT,
			AlphaDecode:  p.config.AlphaTPOT,
			TierWeights:  tierWeights,
		},
		Candidate: placementevalclient.RequestSpec{
			ID: candID, ArrivalTime: 0,
			InputTokens: inTok, OutputTokens: outTok, SLOClass: cls,
		},
		Future: futSpecs,
		Pods:   pods,
	}

	resp, err := p.evalClient.Evaluate(ctx, evalReq)
	if err != nil {
		logger.V(logutil.DEBUG).Info("placement-eval failed; leaving FutureCost unset", "err", err.Error())
		return
	}
	byPod := make(map[string]float64, len(resp.Results))
	for _, r := range resp.Results {
		byPod[r.PodID] = r.VH
	}
	for i, ep := range endpoints {
		if attrs[i] == nil || ep == nil || ep.GetMetadata() == nil {
			continue
		}
		if vh, ok := byPod[ep.GetMetadata().ID.String()]; ok {
			attrs[i].WithFutureCost(vh)
			ep.Put(p.slaRiskDataKey, attrs[i])
		}
	}
}

func (p *Plugin) buildFutureWindow() []slaaware.FutureArrival {
	switch p.config.FutureWorkloadMode {
	case futureModeSynthetic:
		return slaaware.BuildSyntheticWindow(
			p.config.HorizonUs, p.config.SyntheticRatePerSec,
			p.config.SyntheticPromptTokens, p.config.SyntheticOutputTokens,
			p.config.SyntheticSLOClass, "syn",
		)
	default:
		return slaaware.BuildRecentHistoryWindow(p.history.Snapshot(), p.config.HorizonUs, "hist")
	}
}

// Produces declares the SLARiskInfo attribute.
func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.slaRiskDataKey: attrslarisk.SLARiskInfo{}}
}

// Consumes requires latency predictions.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			p.latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{},
		},
	}
}

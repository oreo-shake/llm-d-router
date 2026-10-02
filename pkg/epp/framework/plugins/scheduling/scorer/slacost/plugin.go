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

// Package slacost scores endpoints by immediate SLA risk C_now, soft violation
// C_violation when the feasible set is empty, and optional λ·V_H when
// SLARiskInfo carries FutureCost from the placement-eval sidecar.
package slacost

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrslarisk "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/slarisk"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/slaaware"
)

const (
	// PluginType is the registered type name.
	PluginType = "sla-cost-scorer"
)

var _ fwksched.Scorer = &Plugin{}

// Config is the JSON parameters for sla-cost-scorer.
type Config struct {
	slaaware.TierTable
	// Lambda weights the future-cost term V_H. Zero disables C_future (default).
	Lambda float64 `json:"lambda,omitempty"`

	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	SLARiskInfoProducerName           string `json:"slaRiskInfoProducerName,omitempty"`
}

// DefaultConfig is used when parameters are omitted.
var DefaultConfig = Config{}

// Plugin implements the SLA cost objective.
type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	latencyPredictionInfoDataKey fwkplugin.DataKey
	slaRiskDataKey               fwkplugin.DataKey
}

// NewPlugin constructs a Plugin for tests.
func NewPlugin(config Config) *Plugin {
	_ = config.Normalize()
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: PluginType},
		config:                       config,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		slaRiskDataKey:               attrslarisk.SLARiskInfoDataKey.WithNonEmptyProducerName(config.SLARiskInfoProducerName),
	}
}

// Factory constructs the scorer.
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
	if config.Lambda < 0 {
		return nil, fmt.Errorf("lambda must be >= 0, got %f", config.Lambda)
	}
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		slaRiskDataKey:               attrslarisk.SLARiskInfoDataKey.WithNonEmptyProducerName(config.SLARiskInfoProducerName),
	}, nil
}

// TypedName returns the plugin identity.
func (s *Plugin) TypedName() fwkplugin.TypedName { return s.typedName }

// Category classifies this scorer.
func (s *Plugin) Category() fwksched.ScorerCategory { return fwksched.Balance }

type cand struct {
	ep       fwksched.Endpoint
	risk     slaaware.EndpointRisk
	attr     *attrslarisk.SLARiskInfo
	feasible bool
}

// Score returns [0,1] scores for C_now (+ λ V_H when FutureCost is set).
// C_now = α_P (1 - F_P) + α_D (1 - F_D). Missing V_H with lambda>0 falls back to C_now only.
func (s *Plugin) Score(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	logger := log.FromContext(ctx)
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		scores[ep] = 0
	}
	if len(endpoints) == 0 {
		return scores
	}

	headers := map[string]string{}
	if request != nil {
		headers = request.Headers
	}
	slo, err := s.config.ResolveRequestSLO(headers)
	if err != nil {
		logger.V(logutil.DEBUG).Info("SLACostScorer: SLO resolve failed", "err", err.Error())
		return scores
	}

	cands := make([]cand, 0, len(endpoints))
	anyFeasible := false
	for _, ep := range endpoints {
		risk, attr, ok := s.endpointRisk(ep, slo)
		if !ok {
			continue
		}
		c := cand{ep: ep, risk: risk, attr: attr, feasible: risk.Feasible}
		if c.feasible {
			anyFeasible = true
		}
		cands = append(cands, c)
	}
	if len(cands) == 0 {
		logger.V(logutil.DEBUG).Info("SLACostScorer: no risk data")
		return scores
	}

	for _, c := range cands {
		var cost float64
		if anyFeasible {
			if !c.feasible {
				scores[c.ep] = 0
				continue
			}
			cost = slaaware.CostNow(c.risk, slo)
		} else {
			cost = slaaware.CostViolation(c.risk, slo)
		}
		if s.config.Lambda > 0 && c.attr != nil && c.attr.HasFuture() {
			cost += s.config.Lambda * c.attr.FutureCost()
		}
		scores[c.ep] = slaaware.ScoreFromCost(cost)
	}
	return scores
}

func (s *Plugin) endpointRisk(ep fwksched.Endpoint, slo slaaware.RequestSLO) (slaaware.EndpointRisk, *attrslarisk.SLARiskInfo, bool) {
	if raw, ok := ep.Get(s.slaRiskDataKey); ok {
		info := raw.(*attrslarisk.SLARiskInfo)
		return slaaware.EndpointRisk{
			TTFTPred:      info.TTFTPred(),
			TPOTPred:      info.TPOTPred(),
			TTFTSurvival:  info.TTFTSurvival(),
			TPOTSurvival:  info.TPOTSurvival(),
			Feasible:      info.Feasible(),
			Source:        info.Source(),
			HasPrediction: true,
		}, info, true
	}
	if raw, ok := ep.Get(s.latencyPredictionInfoDataKey); ok {
		info := raw.(*attrlatency.LatencyPredictionInfo)
		return slaaware.RiskFromQuantile(info.TTFT(), info.TPOT(), slo), nil, true
	}
	return slaaware.EndpointRisk{}, nil, false
}

// Consumes declares optional risk / latency attributes.
func (s *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			s.slaRiskDataKey:               attrslarisk.SLARiskInfo{},
			s.latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{},
		},
	}
}

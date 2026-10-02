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

// Package slarisk implements the ε-risk feasible-set filter F_i.
// When the feasible set is empty, all candidates are passed through so the
// sla-cost-scorer can apply soft violation ranking.
package slarisk

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
	PluginType = "sla-risk-filter"
)

var _ fwksched.Filter = &Plugin{}

// Config is the JSON parameters for sla-risk-filter.
type Config struct {
	slaaware.TierTable
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	SLARiskInfoProducerName           string `json:"slaRiskInfoProducerName,omitempty"`
}

// DefaultConfig is used when parameters are omitted.
var DefaultConfig = Config{}

// Plugin keeps endpoints in the ε-risk feasible set when non-empty.
type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	latencyPredictionInfoDataKey fwkplugin.DataKey
	slaRiskDataKey               fwkplugin.DataKey
}

// Factory constructs the filter.
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
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		slaRiskDataKey:               attrslarisk.SLARiskInfoDataKey.WithNonEmptyProducerName(config.SLARiskInfoProducerName),
	}, nil
}

// TypedName returns the plugin identity.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// Filter returns the feasible set F_i, or all endpoints when F_i is empty /
// when no risk/prediction data is available.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	if len(endpoints) <= 1 {
		return endpoints
	}

	headers := map[string]string{}
	if request != nil {
		headers = request.Headers
	}
	slo, err := p.config.ResolveRequestSLO(headers)
	if err != nil {
		logger.V(logutil.DEBUG).Info("SLARiskFilter: SLO resolve failed, keeping all", "err", err.Error())
		return endpoints
	}

	var feasible, infeasible, unknown []fwksched.Endpoint
	for _, ep := range endpoints {
		risk, ok := p.endpointRisk(ep, slo)
		if !ok {
			unknown = append(unknown, ep)
			continue
		}
		if risk.Feasible {
			feasible = append(feasible, ep)
		} else {
			infeasible = append(infeasible, ep)
		}
	}

	if len(feasible) == 0 && len(infeasible) == 0 {
		logger.V(logutil.DEBUG).Info("SLARiskFilter: no risk data, keeping all", "total", len(endpoints))
		return endpoints
	}

	if len(feasible) == 0 {
		// Empty F_i: pass everyone (including unknown) for soft violation scoring.
		logger.V(logutil.DEBUG).Info("SLARiskFilter: empty feasible set, passthrough",
			"infeasible", len(infeasible), "unknown", len(unknown))
		return endpoints
	}

	logger.V(logutil.DEBUG).Info("SLARiskFilter: selecting feasible set",
		"feasible", len(feasible), "infeasible", len(infeasible), "unknown", len(unknown))
	return feasible
}

func (p *Plugin) endpointRisk(ep fwksched.Endpoint, slo slaaware.RequestSLO) (slaaware.EndpointRisk, bool) {
	if raw, ok := ep.Get(p.slaRiskDataKey); ok {
		info := raw.(*attrslarisk.SLARiskInfo)
		return slaaware.EndpointRisk{
			TTFTPred:      info.TTFTPred(),
			TPOTPred:      info.TPOTPred(),
			TTFTSurvival:  info.TTFTSurvival(),
			TPOTSurvival:  info.TPOTSurvival(),
			Feasible:      info.Feasible(),
			Source:        info.Source(),
			HasPrediction: true,
		}, true
	}
	if raw, ok := ep.Get(p.latencyPredictionInfoDataKey); ok {
		info := raw.(*attrlatency.LatencyPredictionInfo)
		return slaaware.RiskFromQuantile(info.TTFT(), info.TPOT(), slo), true
	}
	return slaaware.EndpointRisk{}, false
}

// Consumes declares optional risk / latency attributes.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			p.slaRiskDataKey:               attrslarisk.SLARiskInfo{},
			p.latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{},
		},
	}
}

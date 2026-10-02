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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrslarisk "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/slarisk"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func makeEP(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "default"}},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)
}

func setLatency(ep fwksched.Endpoint, ttft, tpot float64) {
	ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(
		true, true, 0, 0, ttft, tpot, 0))
}

func setRisk(ep fwksched.Endpoint, feasible bool, ttftSurv, tpotSurv float64) {
	ep.Put(attrslarisk.SLARiskInfoDataKey, attrslarisk.NewSLARiskInfo(
		100, 10, ttftSurv, tpotSurv, feasible, "cdf"))
}

func newPlugin() *Plugin {
	cfg := DefaultConfig
	_ = cfg.Normalize()
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: "test"},
		config:                       cfg,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey,
		slaRiskDataKey:               attrslarisk.SLARiskInfoDataKey,
	}
}

func reqWithSLO(ttft, tpot string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.TTFTSLOHeaderKey: ttft,
		metadata.TPOTSLOHeaderKey: tpot,
	}}
}

func TestFilter_KeepsFeasibleOnly(t *testing.T) {
	p := newPlugin()
	ok := makeEP("ok")
	bad := makeEP("bad")
	setLatency(ok, 50, 10)
	setLatency(bad, 300, 10)
	out := p.Filter(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{ok, bad})
	require.Len(t, out, 1)
	assert.Equal(t, "ok", out[0].GetMetadata().ID.Name)
}

func TestFilter_EmptyFeasiblePassthrough(t *testing.T) {
	p := newPlugin()
	a := makeEP("a")
	b := makeEP("b")
	setLatency(a, 300, 10)
	setLatency(b, 400, 10)
	out := p.Filter(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, b})
	assert.Len(t, out, 2)
}

func TestFilter_MixedTTFTOkTPOTFail(t *testing.T) {
	p := newPlugin()
	a := makeEP("a")
	setLatency(a, 50, 80) // TPOT fails
	out := p.Filter(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, makeEP("b")})
	// a infeasible, b unknown → empty feasible → passthrough all
	assert.Len(t, out, 2)
}

func TestFilter_PrefersSLARiskAttribute(t *testing.T) {
	p := newPlugin()
	a := makeEP("a")
	b := makeEP("b")
	setRisk(a, true, 0.05, 0.05)
	setRisk(b, false, 0.5, 0.05)
	out := p.Filter(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, b})
	require.Len(t, out, 1)
	assert.Equal(t, "a", out[0].GetMetadata().ID.Name)
}

func TestFactory(t *testing.T) {
	pl, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, pl.TypedName().Type)
}

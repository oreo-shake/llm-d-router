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

package sladistribution

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

func TestProduce_QuantileAndCDFModes(t *testing.T) {
	for _, mode := range []string{modeQuantile, modeCDF} {
		t.Run(mode, func(t *testing.T) {
			raw := []byte(`{"mode":"` + mode + `"}`)
			pl, err := Factory("sla-distribution-producer", fwkplugin.StrictDecoder(raw), nil)
			require.NoError(t, err)
			p := pl.(*Plugin)

			ep := fwksched.NewEndpoint(
				&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "p1"}},
				&fwkdl.Metrics{WaitingQueueSize: 3, KVCacheUsagePercent: 0.4},
				fwkdl.NewAttributes(),
			)
			ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(
				true, true, 0, 0, 80, 15, 0))

			req := &fwksched.InferenceRequest{Headers: map[string]string{
				metadata.TTFTSLOHeaderKey: "200",
				metadata.TPOTSLOHeaderKey: "50",
			}}
			require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))

			rawAttr, ok := ep.Get(p.slaRiskDataKey)
			require.True(t, ok)
			info := rawAttr.(*attrslarisk.SLARiskInfo)
			assert.True(t, info.Feasible())
			assert.Equal(t, mode, info.Source())
			assert.False(t, info.HasFuture())
		})
	}
}

func TestProduce_LogNormalSigmaOverridesLogistic(t *testing.T) {
	raw := []byte(`{"mode":"cdf","tiers":[{"name":"gold","ttftSloMs":200,"tpotSloMs":50,"epsilonTtft":0.5,"epsilonTpot":0.5}]}`)
	pl, err := Factory("sla-distribution-producer", fwkplugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	p := pl.(*Plugin)

	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "p1"}},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)
	ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(
		true, true, 0, 0, 200, 50, 0).WithLogSigma(0.4, 0.2))

	req := &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.TTFTSLOHeaderKey: "200",
		metadata.TPOTSLOHeaderKey: "50",
	}}
	require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
	rawAttr, ok := ep.Get(p.slaRiskDataKey)
	require.True(t, ok)
	info := rawAttr.(*attrslarisk.SLARiskInfo)
	assert.Equal(t, "cdf", info.Source())
	assert.InDelta(t, 0.5, info.TTFTSurvival(), 0.05)
}

func TestProduce_DistReadyAFT(t *testing.T) {
	raw := []byte(`{"mode":"cdf","survivalModel":"aft","tiers":[{"name":"gold","ttftSloMs":200,"tpotSloMs":50,"epsilonTtft":0.5,"epsilonTpot":0.5}]}`)
	pl, err := Factory("sla-distribution-producer", fwkplugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	p := pl.(*Plugin)

	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "p1"}},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)
	ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(
		true, true, 0, 0, 80, 15, 0).WithDist(0.9, 0.2, 0.3, 0.1, 40, 5, true))

	req := &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.TTFTSLOHeaderKey: "200",
		metadata.TPOTSLOHeaderKey: "50",
	}}
	require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
	rawAttr, ok := ep.Get(p.slaRiskDataKey)
	require.True(t, ok)
	info := rawAttr.(*attrslarisk.SLARiskInfo)
	assert.Equal(t, "aft", info.Source())
	assert.InDelta(t, 0.3, info.TTFTSurvival(), 1e-9)
	assert.InDelta(t, 0.1, info.TPOTSurvival(), 1e-9)
	assert.InDelta(t, 40, info.TTFTOverrun(), 1e-9)
}

func TestFactoryInvalidMode(t *testing.T) {
	_, err := Factory("t", fwkplugin.StrictDecoder([]byte(`{"mode":"nope"}`)), nil)
	assert.Error(t, err)
}

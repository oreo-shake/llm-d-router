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

package slacost

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
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
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		fwkdl.NewAttributes(),
	)
}

func setLatency(ep fwksched.Endpoint, ttft, tpot float64) {
	ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(
		true, true, 0, 0, ttft, tpot, 0))
}

func reqWithSLO(ttft, tpot string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.TTFTSLOHeaderKey: ttft,
		metadata.TPOTSLOHeaderKey: tpot,
	}}
}

func TestScore_PrefersLowerImmediateRisk(t *testing.T) {
	s := NewPlugin(DefaultConfig)
	better := makeEP("better")
	worse := makeEP("worse")
	setLatency(better, 40, 10)
	setLatency(worse, 150, 10)
	scores := s.Score(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{better, worse})
	assert.Greater(t, scores[better], scores[worse])
}

func TestScore_ViolationWhenNoneFeasible(t *testing.T) {
	s := NewPlugin(DefaultConfig)
	a := makeEP("a")
	b := makeEP("b")
	setLatency(a, 250, 10)
	setLatency(b, 400, 10)
	scores := s.Score(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, b})
	assert.Greater(t, scores[a], scores[b])
}

func TestScore_ZerosInfeasibleWhenFeasibleExists(t *testing.T) {
	s := NewPlugin(DefaultConfig)
	ok := makeEP("ok")
	bad := makeEP("bad")
	setLatency(ok, 50, 10)
	setLatency(bad, 300, 10)
	scores := s.Score(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{ok, bad})
	assert.Greater(t, scores[ok], 0.0)
	assert.Equal(t, 0.0, scores[bad])
}

func TestScore_LambdaUsesSidecarVH(t *testing.T) {
	s := NewPlugin(Config{Lambda: 2})
	a := makeEP("a")
	b := makeEP("b")
	ra := attrslarisk.NewSLARiskInfo(50, 10, 0.1, 0.1, true, "cdf").WithFutureCost(0.1)
	rb := attrslarisk.NewSLARiskInfo(50, 10, 0.1, 0.1, true, "cdf").WithFutureCost(5)
	a.Put(attrslarisk.SLARiskInfoDataKey, ra)
	b.Put(attrslarisk.SLARiskInfoDataKey, rb)
	scores := s.Score(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, b})
	assert.Greater(t, scores[a], scores[b])
}

func TestScore_LambdaWithoutFutureFallsBackToNow(t *testing.T) {
	s := NewPlugin(Config{Lambda: 10})
	a := makeEP("a")
	b := makeEP("b")
	setLatency(a, 40, 10)
	setLatency(b, 150, 10)
	scores := s.Score(context.Background(), reqWithSLO("200", "50"), []fwksched.Endpoint{a, b})
	assert.Greater(t, scores[a], scores[b])
}

func TestFactory_RejectsNegativeLambda(t *testing.T) {
	_, err := Factory("t", fwkplugin.StrictDecoder([]byte(`{"lambda": -1}`)), nil)
	assert.Error(t, err)
}

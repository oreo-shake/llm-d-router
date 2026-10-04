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

package predictedlatency

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrefixShapeGPUCPUUnmatched(t *testing.T) {
	info := attrprefix.NewPrefixCacheMatchInfo(10, 16, 16).
		WithCachedBlockCount(10).
		WithCachedBlocksByTier(map[string]int{"gpu": 6, "cpu": 10})
	s := prefixShapeFromInfo(info)
	assert.Equal(t, 16, s.n)
	assert.Equal(t, 10, s.m)
	assert.Equal(t, 6, s.mGPU)
	assert.Equal(t, 4, s.mCPU)
	assert.Equal(t, 6, s.unmatchedBlocks())

	shape := buildLatencyShape(s, &fwkdl.Metrics{WaitingQueueSize: 2, KVCacheUsagePercent: 0.25}, 100, 80, 50, 20, 200, 40)
	assert.Equal(t, 6*16, shape.CachedGPUTokens)
	assert.Equal(t, 4*16, shape.CachedCPUTokens)
	assert.Equal(t, 6*16, shape.UnmatchedTokens)
	assert.InDelta(t, 16*16-10*10, shape.UnmatchedQuadraticBlocks, 1e-9)
	assert.Equal(t, 200, shape.WaitingTokens)
	assert.InDelta(t, 0.75, shape.KVFreeFraction, 1e-9)
	assert.Equal(t, int64(80), shape.UncachedRequestTokens)
	assert.Equal(t, 200.0, shape.SLOTTFTMs)
}

func TestResidualSigmaWarmsThenShrinks(t *testing.T) {
	tr := newResidualTracker()
	assert.InDelta(t, residualWarmSigma, tr.TTFTSigma(), 1e-9)
	for i := 0; i < 20; i++ {
		tr.ObserveTTFT("p0", 100, 110)
	}
	assert.Less(t, tr.TTFTSigma(), residualWarmSigma)
	ttft, _ := tr.PodEWMA("p0")
	assert.Greater(t, ttft, 0.0)
	assert.Equal(t, minPositiveMs, clampPositivePred(0, 0))
	assert.Equal(t, 50.0, clampPositivePred(0, 50))
	assert.Equal(t, 12.0, clampPositivePred(12, 50))
	assert.Greater(t, tr.TTFTMu(), 0.0)
	assert.InDelta(t, 100*math.Exp(tr.TTFTMu()), calibratePred(100, tr.TTFTMu()), 1e-6)
}

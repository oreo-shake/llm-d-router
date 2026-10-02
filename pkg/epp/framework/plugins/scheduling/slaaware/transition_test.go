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

package slaaware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestHistoryBuffer_Replay(t *testing.T) {
	b := NewHistoryBuffer(10, time.Minute)
	base := time.Now()
	b.Add(ArrivalRecord{ArrivedAt: base, InputTokens: 10, OutputTokens: 5, SLOClass: "default"})
	b.Add(ArrivalRecord{ArrivedAt: base.Add(100 * time.Millisecond), InputTokens: 20, OutputTokens: 5, SLOClass: "gold"})
	fut := BuildRecentHistoryWindow(b.Snapshot(), 1_000_000, "h")
	assert.Len(t, fut, 2)
	assert.Equal(t, int64(0), fut[0].ArrivalTimeUs)
	assert.Greater(t, fut[1].ArrivalTimeUs, int64(0))
}

func TestBuildSyntheticWindow(t *testing.T) {
	fut := BuildSyntheticWindow(1_000_000, 10, 64, 16, "default", "s")
	assert.GreaterOrEqual(t, len(fut), 10)
	assert.LessOrEqual(t, fut[len(fut)-1].ArrivalTimeUs, int64(1_000_000))
}

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

// Package slaaware holds shared SLA-tier contracts and risk/cost math used by
// the sla-risk-filter, sla-cost-scorer, and sla-distribution-producer plugins.
package slaaware

import (
	"sync"
	"time"
)

// ArrivalRecord is one observed request used to build a recent-history future window.
type ArrivalRecord struct {
	ArrivedAt    time.Time
	InputTokens  int
	OutputTokens int
	SLOClass     string
}

// HistoryBuffer is a concurrent ring of recent arrivals.
type HistoryBuffer struct {
	mu      sync.Mutex
	maxN    int
	maxAge  time.Duration
	records []ArrivalRecord
}

// NewHistoryBuffer creates a buffer retaining at most maxN records within maxAge.
func NewHistoryBuffer(maxN int, maxAge time.Duration) *HistoryBuffer {
	if maxN <= 0 {
		maxN = 256
	}
	if maxAge <= 0 {
		maxAge = 30 * time.Second
	}
	return &HistoryBuffer{maxN: maxN, maxAge: maxAge}
}

// Add appends a record and prunes stale / overflow entries.
func (b *HistoryBuffer) Add(r ArrivalRecord) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if r.ArrivedAt.IsZero() {
		r.ArrivedAt = time.Now()
	}
	b.records = append(b.records, r)
	b.pruneLocked(time.Now())
}

// Snapshot returns a copy of retained records (oldest first).
func (b *HistoryBuffer) Snapshot() []ArrivalRecord {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pruneLocked(time.Now())
	out := make([]ArrivalRecord, len(b.records))
	copy(out, b.records)
	return out
}

// Len returns the number of retained records.
func (b *HistoryBuffer) Len() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.records)
}

func (b *HistoryBuffer) pruneLocked(now time.Time) {
	cutoff := now.Add(-b.maxAge)
	i := 0
	for i < len(b.records) && b.records[i].ArrivedAt.Before(cutoff) {
		i++
	}
	if i > 0 {
		b.records = append([]ArrivalRecord(nil), b.records[i:]...)
	}
	if len(b.records) > b.maxN {
		b.records = append([]ArrivalRecord(nil), b.records[len(b.records)-b.maxN:]...)
	}
}

// ReplayOntoHorizon shifts recent-history relative deltas onto [t0, t0+horizon].
// Returns wire-friendly arrivals with ArrivalTimeUs relative to t0Us.
type FutureArrival struct {
	ID             string
	ArrivalTimeUs  int64
	InputTokens    int
	OutputTokens   int
	SLOClass       string
}

// BuildRecentHistoryWindow replays history onto [0, horizonUs] using relative gaps.
func BuildRecentHistoryWindow(history []ArrivalRecord, horizonUs int64, idPrefix string) []FutureArrival {
	if horizonUs <= 0 || len(history) == 0 {
		return nil
	}
	out := make([]FutureArrival, 0, len(history))
	base := history[0].ArrivedAt
	for i, r := range history {
		delta := r.ArrivedAt.Sub(base).Microseconds()
		if delta < 0 {
			delta = 0
		}
		if delta > horizonUs {
			break
		}
		cls := r.SLOClass
		if cls == "" {
			cls = "default"
		}
		out = append(out, FutureArrival{
			ID:            idPrefix + "-h" + itoa(i),
			ArrivalTimeUs: delta,
			InputTokens:   r.InputTokens,
			OutputTokens:  maxInt(1, r.OutputTokens),
			SLOClass:      cls,
		})
	}
	return out
}

// BuildSyntheticWindow creates constant-rate arrivals on [0, horizonUs].
func BuildSyntheticWindow(horizonUs int64, ratePerSec float64, promptTokens, outputTokens int, sloClass, idPrefix string) []FutureArrival {
	if horizonUs <= 0 || ratePerSec <= 0 {
		return nil
	}
	if promptTokens <= 0 {
		promptTokens = 128
	}
	if outputTokens <= 0 {
		outputTokens = 32
	}
	if sloClass == "" {
		sloClass = "default"
	}
	intervalUs := int64(1e6 / ratePerSec)
	if intervalUs < 1 {
		intervalUs = 1
	}
	var out []FutureArrival
	for t, i := int64(0), 0; t <= horizonUs; t, i = t+intervalUs, i+1 {
		out = append(out, FutureArrival{
			ID:            idPrefix + "-s" + itoa(i),
			ArrivalTimeUs: t,
			InputTokens:   promptTokens,
			OutputTokens:  outputTokens,
			SLOClass:      sloClass,
		})
	}
	return out
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

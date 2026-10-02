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

package placementevalclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

const (
	defaultURL     = "http://localhost:9091"
	defaultTimeout = 50 * time.Millisecond
	envURL         = "PLACEMENT_EVAL_URL"
	envTimeoutMs   = "PLACEMENT_EVAL_TIMEOUT_MS"
)

// Config holds client settings.
type Config struct {
	URL     string
	Timeout time.Duration
}

// ConfigFromEnv builds Config from environment variables.
func ConfigFromEnv() Config {
	cfg := Config{URL: defaultURL, Timeout: defaultTimeout}
	if v := os.Getenv(envURL); v != "" {
		cfg.URL = v
	}
	if v := os.Getenv(envTimeoutMs); v != "" {
		if ms, err := strconv.Atoi(v); err == nil && ms > 0 {
			cfg.Timeout = time.Duration(ms) * time.Millisecond
		}
	}
	return cfg
}

// CostWeights is the wire form matching BLIS placement-eval.
type CostWeights struct {
	AlphaPrefill float64            `json:"alpha_prefill,omitempty"`
	AlphaDecode  float64            `json:"alpha_decode,omitempty"`
	TierWeights  map[string]float64 `json:"tier_weights,omitempty"`
}

// RequestSpec is one candidate/future request.
type RequestSpec struct {
	ID           string `json:"id"`
	ArrivalTime  int64  `json:"arrival_time_us"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	SLOClass     string `json:"slo_class,omitempty"`
}

// PodSpec is a hydrate observation.
type PodSpec struct {
	PodID               string  `json:"pod_id,omitempty"`
	ClockUs             int64   `json:"clock_us"`
	QueueDepth          int     `json:"queue_depth"`
	BatchSize           int     `json:"batch_size"`
	KVUtilization       float64 `json:"kv_utilization"`
	MaxNumSeqs          int     `json:"max_num_seqs"`
	MaxNumBatchedTokens int     `json:"max_num_batched_tokens"`
	TotalKVBlocks       int     `json:"total_kv_blocks"`
	BlockSizeTokens     int     `json:"block_size_tokens"`
	Seed                int64   `json:"seed,omitempty"`
}

// SLODimTargets mirrors BLIS workload.SLODimTargets JSON.
type SLODimTargets struct {
	TTFTMs float64 `json:"ttft_ms,omitempty"`
	ITLMs  float64 `json:"itl_ms,omitempty"`
	E2EMs  float64 `json:"e2e_ms,omitempty"`
}

// EvaluateRequest is POST /v1/placement/evaluate body.
type EvaluateRequest struct {
	HorizonUs   int64                     `json:"horizon_us"`
	SLOTargets  map[string]SLODimTargets  `json:"slo_targets"`
	CostWeights CostWeights               `json:"cost_weights,omitempty"`
	Candidate   RequestSpec               `json:"candidate"`
	Future      []RequestSpec             `json:"future,omitempty"`
	Pods        []PodSpec                 `json:"pods"`
}

// TierBreakdown is per-tier V_H contribution.
type TierBreakdown struct {
	TTFTViolations int     `json:"ttft_violations"`
	TPOTViolations int     `json:"tpot_violations"`
	Weight         float64 `json:"weight"`
	TierCost       float64 `json:"tier_cost"`
}

// PodResult is one pod's eval result.
type PodResult struct {
	PodID      string                   `json:"pod_id"`
	VH         float64                  `json:"v_h"`
	PerTier    map[string]TierBreakdown `json:"per_tier,omitempty"`
	Attainment float64                  `json:"attainment"`
	CompletedN int                      `json:"completed_n"`
}

// EvaluateResponse is the sidecar response.
type EvaluateResponse struct {
	Results []PodResult `json:"results"`
}

// Client talks to the BLIS placement-eval sidecar.
type Client struct {
	cfg    Config
	http   *http.Client
}

// New creates a Client.
func New(cfg Config) *Client {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}
	return &Client{
		cfg:  cfg,
		http: &http.Client{Timeout: cfg.Timeout},
	}
}

// Evaluate posts a bulk placement eval.
func (c *Client) Evaluate(ctx context.Context, req EvaluateRequest) (*EvaluateResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.URL+"/v1/placement/evaluate", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("placement-eval status %d: %s", resp.StatusCode, string(raw))
	}
	var out EvaluateResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

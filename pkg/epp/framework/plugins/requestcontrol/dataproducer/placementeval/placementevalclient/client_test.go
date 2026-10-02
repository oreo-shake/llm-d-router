/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    15|Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package placementevalclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestClient_Evaluate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/placement/evaluate" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var req EvaluateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.CostWeights.AlphaPrefill != 1 {
			t.Errorf("alpha_prefill = %v", req.CostWeights.AlphaPrefill)
		}
		_ = json.NewEncoder(w).Encode(EvaluateResponse{
			Results: []PodResult{{PodID: "p0", VH: 2.5, CompletedN: 3}},
		})
	}))
	defer srv.Close()

	c := New(Config{URL: srv.URL, Timeout: time.Second})
	resp, err := c.Evaluate(context.Background(), EvaluateRequest{
		HorizonUs:   1e6,
		CostWeights: CostWeights{AlphaPrefill: 1, AlphaDecode: 1},
		Candidate:   RequestSpec{ID: "c0", InputTokens: 10, OutputTokens: 5},
		Pods:        []PodSpec{{PodID: "p0"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 || resp.Results[0].VH != 2.5 {
		t.Fatalf("got %+v", resp)
	}
}

func TestConfigFromEnv_Defaults(t *testing.T) {
	t.Setenv(envURL, "")
	t.Setenv(envTimeoutMs, "")
	cfg := ConfigFromEnv()
	if cfg.URL != defaultURL {
		t.Errorf("URL = %s", cfg.URL)
	}
	if cfg.Timeout != defaultTimeout {
		t.Errorf("Timeout = %v", cfg.Timeout)
	}
}

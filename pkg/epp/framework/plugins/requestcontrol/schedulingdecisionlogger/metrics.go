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

package schedulingdecisionlogger

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	labelExperiment = "experiment"
	labelCfMxW      = "cf_mxw"
	labelAgree      = "agree"
	labelDisagree   = "disagree"
	labelUnknown    = "unknown"
)

var (
	decisionTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "scheduling_decision_total",
			Help: metricsutil.HelpMsgWithStability(
				"Scheduling decisions observed by scheduling-decision-logger, labeled by agreement with the m×w counterfactual.",
				compbasemetrics.ALPHA,
			),
		},
		[]string{labelExperiment, labelCfMxW},
	)

	gateCaseTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "scheduling_decision_gate_case_total",
			Help: metricsutil.HelpMsgWithStability(
				"Scheduling decisions where m×w and IIB counterfactuals disagree on the GPU-tail vs fuller-CPU pattern.",
				compbasemetrics.ALPHA,
			),
		},
		[]string{labelExperiment},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("scheduling decision logger metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{decisionTotal, gateCaseTotal} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register scheduling decision logger metric: %w", err)
		}
	}
	return nil
}

func recordMetrics(experiment string, agreeMxW *bool, gate bool) {
	mxwLabel := labelUnknown
	if agreeMxW != nil {
		if *agreeMxW {
			mxwLabel = labelAgree
		} else {
			mxwLabel = labelDisagree
		}
	}
	decisionTotal.WithLabelValues(experiment, mxwLabel).Inc()
	if gate {
		gateCaseTotal.WithLabelValues(experiment).Inc()
	}
}

# Scheduling Decision Logger

**Type:** `scheduling-decision-logger` (Alpha)

Generic PreRequest plugin that records each scheduling decision for
cross-experiment comparison. It does not change routing. Pair with
`precise-prefix-cache-producer` so candidate tier features are present.

## What is recorded

One JSON object per emitted decision:

| Field | Meaning |
|-------|---------|
| `experiment` | Label from config (e.g. `iib-offline`, `prefix-scalar-0.8`) |
| `chosen` | Live primary-profile target endpoint id |
| `scorer_scores` | Weighted scores from `ScoredCandidates` |
| `candidates[]` | Per pod: `n`, `m`, `m_gpu`, `m_cpu`, `match_blocks` |
| `counterfactual.mxw` | `argmax(match_blocks)` on the same candidate set |
| `counterfactual.iib` | `argmin(Method IIB cost)` when `iibWeights` is set |
| `agree_mxw` / `agree_iib` | Whether `chosen` matches each counterfactual |
| `gate_case` | m×w GPU-tail vs IIB fuller-CPU disagreement pattern |

Metrics (always, independent of JSONL sampling):

- `scheduling_decision_total{experiment,cf_mxw=agree\|disagree\|unknown}`
- `scheduling_decision_gate_case_total{experiment}`

Audit I/O errors never fail the request.

## Configuration

List the plugin under `plugins:` (auto-wired as PreRequest). Requires
`--allow-experimental-plugins`.

```yaml
plugins:
  - name: decision-logger
    type: scheduling-decision-logger
    parameters:
      experiment: iib-offline
      prefixMatchInfoProducerName: precise-prefix-cache-producer
      outputPath: /var/log/epp/decisions.jsonl   # optional; empty = logs only
      logAgrees: false                          # default: emit disagreements only
      sampleRate: 1.0                           # applies only when logAgrees=true
      iibWeights:                               # optional; enables counterfactual.iib
        gpu: 0
        cpu: 5.3417156e-05
        unmatchedLinear: 0.00052750113
        unmatchedQuadratic: 1.6023985e-07

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: ttft-cost   # or prefix-cache-scorer; logger is independent
        weight: 1
      - pluginRef: max-score-picker
```

Mount `outputPath` via emptyDir or NFS if you need durable JSONL.

## Analysis

From Inference-perf-internal:

```bash
python3 scripts/analyze_scheduling_decisions.py \
  --jsonl reports-iib-decision-audit-<stamp>/decisions.jsonl
```

Within-run gate: m×w disagreement rate under about 5% on an IIB run means
routing rarely differed from `match_blocks` ranking for that workload; stop
retuning IIB coefficients until the candidate set or load changes.

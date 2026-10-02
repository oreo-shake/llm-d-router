# sla-risk-filter

Keeps endpoints in the ε-risk feasible set \(\mathcal{F}_i\): pods whose predicted
TTFT/TPOT risk stays within the request tier budgets. When \(\mathcal{F}_i\) is empty,
all candidates pass through so `sla-cost-scorer` can apply soft violation ranking.

## Inputs

- Prefer `SLARiskInfo` from `sla-distribution-producer` when present.
- Otherwise derive quantile feasibility from `LatencyPredictionInfo` (`Q ≤ S`).

## Request SLO

- `x-llm-d-slo-tier` selects a named tier from `tiers[]`.
- Or `x-llm-d-slo-ttft-ms` / `x-llm-d-slo-tpot-ms` supply thresholds directly.

## Example

```yaml
plugins:
  - type: predicted-latency-producer
  - type: sla-distribution-producer
    parameters:
      mode: quantile
  - type: sla-risk-filter
  - type: sla-cost-scorer
  - type: weighted-random-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: predicted-latency-producer
      - pluginRef: sla-distribution-producer
      - pluginRef: sla-risk-filter
      - pluginRef: sla-cost-scorer
      - pluginRef: weighted-random-picker
```

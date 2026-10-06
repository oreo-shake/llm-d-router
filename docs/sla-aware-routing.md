# SLA-Aware Routing

Risk-constrained routing that ranks pods by immediate SLA probability risk and
optional simulated future SLA violation cost \(V_H\).

## Cost definitions

### Immediate (\(C^{now}\))

\[
C_{ij}^{now} = \alpha_P\bigl(1 - F_{\theta_P}(S_i^P\mid\phi_{ij})\bigr)
+ \alpha_D\bigl(1 - F_{\theta_D}(S_i^D\mid\phi_{ij})\bigr)
\]

Implemented by `sla-cost-scorer` via survival probabilities on `SLARiskInfo`
(`alphaTtft` / `alphaTpot`).

### Future (\(C^{future} = V_H\))

\[
V_H^{(c)} = \alpha_P N_c^{P} + \alpha_D N_c^{D},\qquad
C_{ij}^{future} = V_H(X_t^{+ij}) = \sum_c w_c V_H^{(c)}
\]

\(N_c^{P}\) / \(N_c^{D}\) are TTFT / TPOT violation **counts** on the simulated
decision set (candidate ∪ future window) for tier \(c\). Computed by the BLIS
`placement-eval-server` sidecar (`blis placement-eval-server`), **not** by
`1 - SLOAttainment`.

Default `lambda: 0` → \(V_H\) is not used. Set `lambda > 0` and
`enablePlacementEval: true` on `sla-distribution-producer` to attach sidecar
`v_h` as `FutureCost`.

## Contracts

### Feature vector \(\phi_{ij}\)

| Feature | Source |
|---------|--------|
| Residual-calibrated TTFT / TPOT and log-residual sigma | `LatencyPredictionInfo` (point pred * exp(mu) after warm-up) |
| Survival P(L>S), feasible flag | `SLARiskInfo`. In `cdf` mode: online AFT (`survivalModel: aft`, default) or exceedance MLP (\(\mathrm{logit}=z_{\mathrm{soft}}+\alpha\tanh(\mathrm{MLP})\); rich \(\phi\), \(\hat L\); local forward) when `DistReady`; otherwise lognormal wrap of the sidecar point pred |
| GPU/CPU cached tokens, unmatched tokens, unmatched quadratic (n^2-m^2 blocks) | `predicted-latency-producer` from `PrefixCacheMatchInfo` |
| Waiting tokens, KV free fraction, uncached request tokens, per-pod EWMA | `predicted-latency-producer` |
| Active reload jobs (in-flight CPU→GPU H2D) | `predicted-latency-producer` from scraped `vllm:kv_offload_active_reload_jobs` |
| Future cost V_H | `SLARiskInfo.FutureCost` from placement-eval sidecar |

### Tier table

Per tier \(C\): `(ttftSloMs, tpotSloMs, epsilonTtft, epsilonTpot, weight)`.

Request resolution: `x-llm-d-slo-tier` → named tier; else SLO headers; else first tier.

## Plugin stack

```text
predicted-latency-producer
        ↓
sla-distribution-producer   (quantile|cdf; survivalModel aft|exceedance|lognormal)
        ↓
sla-risk-filter             (F_i; empty → passthrough)
        ↓
sla-cost-scorer             (C_now from P(L>S), or useExpectedOverrun; + λ V_H)
        ↓
picker
```

Alpha plugins require `--allow-experimental-plugins`.

### Example profile (λ = 0)

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: predicted-latency-producer
  - type: sla-distribution-producer
    parameters:
      mode: cdf
      survivalModel: aft
      tiers:
        - name: gold
          ttftSloMs: 200
          tpotSloMs: 40
          epsilonTtft: 0.05
          weight: 2
  - type: sla-risk-filter
  - type: sla-cost-scorer
    parameters:
      lambda: 0
  - type: weighted-random-picker
  - type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: predicted-latency-producer
      - pluginRef: sla-distribution-producer
      - pluginRef: sla-risk-filter
      - pluginRef: sla-cost-scorer
      - pluginRef: weighted-random-picker
```

### Enabling simulated \(V_H\)

**Helm** (co-located BLIS sidecar; sets `PLACEMENT_EVAL_URL` / `PLACEMENT_EVAL_TIMEOUT_MS`):

```yaml
router:
  placementEval:
    enabled: true
    port: 9091
    timeoutMs: "50"
    image:
      registry: ghcr.io/inference-sim
      repository: blis
      tag: latest
```

**Plugins:**

1. Producer: `enablePlacementEval: true`, `futureWorkloadMode: recent-history|synthetic`
2. Scorer: `lambda: 0.5` (or other > 0)

Manual (no Helm): `blis placement-eval-server --listen :9091` and
`PLACEMENT_EVAL_URL=http://localhost:9091`.

On sidecar timeout, `FutureCost` is unset and the scorer uses \(C^{now}\) only.

## Success metric

Goodput (TTFT+TPOT attainment), compared to stock `slo-headroom-tier-filter` + `latency-scorer`.

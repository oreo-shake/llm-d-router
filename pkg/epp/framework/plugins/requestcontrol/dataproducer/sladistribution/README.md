# sla-distribution-producer

Writes `SLARiskInfo` per endpoint from `LatencyPredictionInfo` and the request SLO.

| Mode | Feasibility | Survival |
|------|-------------|----------|
| `quantile` (default) | \(Q \le S\) | logistic map of \((Q,S)\) |
| `cdf` | \(P(L>S) \le \epsilon\) | lognormal from residual-calibrated pred and log-residual sigma; logistic if sigma is 0 |

## Future cost \(V_H\)

Set `enablePlacementEval: true` to call the BLIS `placement-eval-server` sidecar.
`FutureCost` on `SLARiskInfo` is set to:

\[
V_H = \sum_c w_c\,(\alpha_P N_c^{P} + \alpha_D N_c^{D})
\]

Future window modes: `recent-history` (default) or `synthetic`.
URL via `placementEvalUrl` or `PLACEMENT_EVAL_URL` (default `http://localhost:9091`).
Timeouts leave `FutureCost` unset so `sla-cost-scorer` with \(\lambda>0\) falls back to \(C^{now}\) only.

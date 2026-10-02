# sla-cost-scorer

Ranks endpoints by:

\[
C_{ij}^{now} = \alpha_P\bigl(1 - F_{\theta_P}(S_i^P\mid\phi_{ij})\bigr)
+ \alpha_D\bigl(1 - F_{\theta_D}(S_i^D\mid\phi_{ij})\bigr)
\]

and, when `lambda > 0` and `SLARiskInfo` carries `FutureCost` from the placement-eval sidecar:

\[
C = C^{now} + \lambda\, V_H,\quad
V_H = \sum_c w_c\,(\alpha_P N_c^{P} + \alpha_D N_c^{D})
\]

When the feasible set is empty, ranks by soft relative violation \(C^{violation}\).

## Parameters

| Field | Default | Meaning |
|-------|---------|---------|
| `alphaTtft` / `alphaTpot` | 1 | \(\alpha_P\) / \(\alpha_D\) in \(C^{now}\) / \(C^{violation}\) |
| `lambda` | 0 | Weight on sidecar \(V_H\) (off by default) |
| `tiers` | — | Named SLA tiers (thresholds, ε, weights \(w_c\)) |

`1 - attainment` from BLIS is **not** used as \(C^{future}\).

## Score range

Costs map to scores in `(0,1]` via `1/(1+cost)`. Infeasible pods score `0` when any feasible pod exists.
Missing \(V_H\) with \(\lambda>0\) uses \(C^{now}\) only.

# TTFT Cost Scorer (Method IIB)

**Type:** `ttft-cost-scorer`

Scores candidate endpoints by predicted time-to-first-token cost, not by
`matched_blocks * weight`. Pair with `precise-prefix-cache-producer` (so
`CachedBlocksByTier` is populated). For the critical A/B, use this scorer
alone in the scheduling profile (prefix-only).

## Formula

```text
cost  = w_gpu * m_gpu + w_cpu * m_cpu + w1 * (N - m) + w2 * (N^2 - m^2)
score = -cost
```

- `N` = total prompt blocks (`TotalBlocks`)
- `m` = unweighted consecutive match (`CachedBlockCount`)
- `m_gpu`, `m_cpu` = exclusive partition of `m` (GPU preferred when a block
  appears on both tiers)
- `w_gpu` is typically `0`

Optional `normalize: true` maps the score to `1 - cost/cost_cold(N)` in
`[0,1]` without changing ranking.

## Modes

| Mode | Behavior |
|------|----------|
| `live` (default) | Start from `weights`. On each `Score`, read scraped `kv_offload_load_*` endpoint attributes (from `core-metrics-extractor` customMetrics), take the scrape-window delta, and refresh `w_cpu = blockBytes / bytesPerSec`. |
| `offline` | Freeze `weights` from config (paste values from `fit_ttft_cost_weights.py`). Bandwidth updates are ignored. |

Live mode needs `live.blockBytes` set. The default vLLM engine mapping already scrapes `vllm:kv_offload_load_{bytes,time}_total` into attributes `kv_offload_load_bytes` / `kv_offload_load_time`. If you override `engineConfigs` for `vllm`, include those `customMetrics` entries yourself.

## Configuration

```yaml
plugins:
  - name: ttft-cost
    type: ttft-cost-scorer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
      mode: live          # or offline
      normalize: false
      weights:
        gpu: 0
        cpu: 0.0001
        unmatchedLinear: 0.000001
        unmatchedQuadratic: 0.0000000001
      live:
        blockBytes: 2097152   # bytes per KV block (model-specific)
        minSampleBytes: 1048576
        # loadBytesAttributeKey: kv_offload_load_bytes
        # loadTimeAttributeKey: kv_offload_load_time

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: ttft-cost
        weight: 1
      - pluginRef: max-score-picker
```

## Offline fit

From Method I ALL p50 rows (`reports/calibrate-cpu-weight-*/weights.csv`):

```bash
python3 scripts/fit_ttft_cost_weights.py \
  --weights-csv reports/calibrate-cpu-weight-20260828-100914/weights.csv \
  --block-size 16
```

Paste the printed `weights:` block into `mode: offline` config.

## Gate

Unit test: at `N=98304` tokens (`block_size=16`), a full CPU hit must score
above a 96000-token GPU hit (unmatched tail).

For live A/B disagreement rates vs `prefix-cache-scorer` (m×w), enable
`scheduling-decision-logger` (see its README) rather than instrumenting this
scorer.

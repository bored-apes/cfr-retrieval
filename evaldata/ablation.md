# Retrieval ablation

51 judged queries. Intervals are 95% percentile bootstrap over queries.

| Configuration | Recall@100 | nDCG@10 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 only | 0.915 ±0.046 | 0.575 ±0.067 | 0.850 ±0.084 | 34 | 72 |
| dense only | 0.947 ±0.025 | 0.658 ±0.051 | 0.951 ±0.049 | 1 | 2 |
| hybrid (rrf) | 1.000 ±0.000 | 0.701 ±0.053 | 0.967 ±0.042 | 32 | 50 |
| hybrid + rerank | 1.000 ±0.000 | 0.717 ±0.051 | 0.971 ±0.029 | 1802 | 2503 |
| + contextual chunks | 0.994 ±0.008 | 0.717 ±0.046 | 0.980 ±0.029 | 1898 | 2409 |

## Paired comparison vs `bm25 only`

| Configuration | ΔnDCG@10 | 95% CI | Verdict |
|---|---|---|---|
| dense only | +0.083 | [+0.012, +0.155] | significant |
| hybrid (rrf) | +0.125 | [+0.088, +0.165] | significant |
| hybrid + rerank | +0.142 | [+0.081, +0.214] | significant |
| + contextual chunks | +0.142 | [+0.081, +0.209] | significant |

## Incremental - each row vs the row above

The comparison that decides whether a stage is worth keeping.

| Configuration | ΔnDCG@10 | 95% CI | Δp50 | Verdict |
|---|---|---|---|---|
| dense only | +0.083 | [+0.012, +0.155] | -34 ms | earns it |
| hybrid (rrf) | +0.043 | [-0.006, +0.098] | +31 ms | no measurable gain |
| hybrid + rerank | +0.016 | [-0.028, +0.061] | +1770 ms | no measurable gain |
| + contextual chunks | -0.000 | [-0.021, +0.018] | +96 ms | no measurable gain |

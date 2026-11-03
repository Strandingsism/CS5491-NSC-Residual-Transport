# NSC starter inspection — 2026-10-15

## Reproduction

Extracted, unmodified starter: `../../nsc-starter/`.
Run from the project root:

```powershell
& .venv/Scripts/python.exe nsc-starter/check_starter.py
& .venv/Scripts/python.exe nsc-starter/evaluate.py --split development --output analysis/starter_audit/development_results.json
```

Self-check passed: MP numerical checks, reorder controls, resource arithmetic, ranking ties, selection, all 200 configurations, split separation, and data hashes. Development output matches all supplied expected values to 1e-12; the sole exact-value discrepancy is random-selection standard deviation at 20M/top-1 (11.825936547801547 versus 11.825936547801545). Final labels were not parsed or evaluated; the supplied self-check hashes their bytes for integrity.

## What the archive supplies

- Standard-library CPU implementation of original NSC, an illustrative position-weighted score, evaluator, resource metadata and archived outcomes. It contains no model implementation, checkpoints, or training pipeline.
- 200 architectures, 100 development and 100 final. Development uses validation perplexity; final uses test perplexity on different architectures from the same archive. This is not an independent external benchmark.
- All 200 configurations have `pre_lnorm=false`, `weight_init_std=0.02`, `proj_init_std=0.01`, `mem_len=tgt_len=192`, adaptive embeddings, and tied weights. Hidden widths are 64/128/256/512; depths range from 2 to 16. Embedding widths and adaptive-embedding division factors vary.
- NSC sums MP capacities for Q/K/V per head, attention output and two FFN matrices, with fixed sigma=0.02. It excludes LayerNorm, embeddings, biases and vocabulary projection. The meaning and application of `proj_init_std` cannot be established without the original training implementation; do not assume it denotes attention output initialization.
- `student_score.py` supplies two placeholders that both return original NSC; no new method is implemented there.
- Resource component fields overlap; use the reported total for parameter budgets. Reference FLOPs are an analytical proxy, not measured training or inference time.

## Development results

| Score | Spearman rho | Full-panel tau-b | Size-controlled pair tau-b |
|---|---:|---:|---:|
| Original NSC | 0.969745 | 0.861414 | 0.622222 |
| Position example | 0.970069 | 0.865051 | 0.622222 |
| Reported parameters | 0.727873 | 0.575758 | 0.133333 |
| Reference FLOPs | 0.967948 | 0.858471 | 0.622222 |

Size control uses the same hidden width and min/max reported total parameters >= 0.90, yielding 180 overlapping pairs. These are not 180 independent observations; equal aggregate tau does not establish equal predictions. The position example's small overall improvement is not evidence of a causal layer-order effect.

## Implications for the proposal

The starter removes the archive-availability blocker, but cannot by itself validate residual propagation or parameter-update mechanisms: it lacks matched initialization/normalization interventions, training trajectories, seeds, and enough implementation detail to exactly reproduce original training.

Two complementary questions should remain distinct:

1. **Mechanism:** in controlled residual models, does a weight-spectrum sum distinguish parameter-update dynamics under matched architecture and initialization/scaling interventions? Post-LN must be treated explicitly when connecting this study to the archive. A Pre-LN variance-growth argument alone does not apply to these configurations. Input Jacobians and parameter Jacobians remain different objects.
2. **Predictive value:** within the archive, does NSC add information beyond computation and size? The very close FLOPs baseline makes this an essential comparison. Use development-only analysis to choose controls and freeze the score/protocol before final evaluation. Consider projection-only parameter counts, embedding/resource decomposition, and sensitivity to matching thresholds; avoid claiming causality from observational comparisons.

For intuition, log det(I + W^T W) approaches ||W||_F^2 in a sufficiently small-eigenvalue regime, whose expectation under iid fixed-variance entries is proportional to matrix entry count. This motivates examining how much NSC's nonlinear spectral terms contribute beyond size. It is not a verified uniform approximation for all matrices in this archive.

Keep the controlled study as the mechanism test and the starter as secondary predictive validation. Do not invent normalization or initialization interventions by editing a config while reusing its original perplexity label. No student scorer or proposal source was changed during this inspection.

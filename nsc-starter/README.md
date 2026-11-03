# NSC course-project starter — CS5491, 2026/27 Semester A

A CPU-only experiment for studying an architecture heuristic. Python 3.9 or later; standard library only. No installation, API key, model download or training is needed for the core project.

## Run

Unzip the starter, open a terminal in its directory, and run:

```sh
python3 check_starter.py
python3 evaluate.py --split development --output development_results.json
```

Edit `student_score.py`. Implement `score(config)` and a mechanism ablation in `ablation(config)`. Both return a finite scalar; higher predicts better trained performance. The supplied template returns original NSC, so it runs but is not a completed project. Configuration input contains no architecture ID or outcome. Do not read label files or use configuration fingerprints as outcome lookups inside a scorer.

After freezing your method, development choices and code version:

```sh
python3 evaluate.py --split final --output final_results.json
```

Final outcomes are held out by protocol, not hidden by software. Do not tune to them. The evaluator loads only the chosen split's label file. Expect seconds on an ordinary laptop; record your actual hardware and elapsed time. Scoring times in the JSON share a cache, so they are diagnostics, not fair cold-start comparisons.

## Files and schema

- `nsc.py`: MP integral, per-layer and original NSC, an illustrative positional score, and a resource calculator.
- `student_score.py`: your two functions. `--student path/to/module.py` selects another implementation.
- `evaluate.py`: rank correlations, size-controlled pair comparisons, and finite-pool selection.
- `check_starter.py`: numerical identities, convergence, tie handling, resource arithmetic, split and data-integrity checks.
- `data/architectures.json`: 200 IDs mapped to full archived configurations. The scorer uses `d_model`, `n_layer`, `n_head`, and `d_inner`; the last two may be per-layer lists. Hidden widths are 64, 128, 256 and 512; depth ranges from 2 to 16. Other fields are preserved for inspecting architectural assumptions.
- `data/resources.json`: archived parameter fields. Use `total` directly; `nonembedding` and component entries overlap. Never sum every field.
- `data/splits.json`: 100 development and 100 final architectures, assigned within hidden-width groups by a fixed configuration hash, without examining performance.
- `data/development_labels.json`: validation perplexity for development architectures.
- `data/final_labels.json`: test perplexity for final architectures.
- `data/provenance.json`: source attribution, SHA-256 hashes, transformations and limitations.
- `expected_development.json`: reference output with the unedited student template; timing fields are omitted.

## Baseline conventions

The original score includes Q/K/V per head, one attention output projection, and two FFN projections per layer:

`S0 = sum_l [3*H_l*psi(d,d/H_l,0.02) + psi(d,d,0.02) + 2*psi(d,f_l,0.02)]`.

The fixed standard deviation is 0.02. This is the NSC archived-panel convention, not a claim that every field of the training configuration uses that scale. Embeddings, biases, normalization and the vocabulary head are excluded from this score. `psi_mp` implements the MP approximation with deterministic Simpson quadrature after a change of variable; it does not sample weights. Doubling the default 512 integration steps is checked, and two archived original-NSC values are cross-checked. No new MP derivation is required from students.

`position_example` assigns layer l (starting at 1) weight l/L. It is a teaching example that makes the role of aggregation visible; it is not a validated capacity model. Merely submitting this example is insufficient. Develop and justify one modification, or a substantive controlled investigation of an existing mechanism, with its ablation and limitations. Publication-level novelty and improved benchmark scores are not required.

For a two-layer example with d=128, H=[4,4] and FFN widths [256,512]:

| Configuration | Original NSC | Position example |
|---|---:|---:|
| FFN [256,512] | 121.820169804 | 96.957958136 |
| FFN [512,256] | 121.820169804 | 85.772296570 |

Both have 327,680 projection weights and 524,288 reference MACs per query token. There are no trained labels for this toy pair: it checks score sensitivity, not predictive validity.

## Resource convention

Projection parameters are `P_proj=sum_l(4*d^2+2*d*f_l)`. Reference MACs per query token are `P_proj+2*L*K*d`, counting the QK and AV products at fixed key length K=384 (192 current plus 192 cached states). Reference FLOPs are twice MACs. This dense-operation reference omits embeddings, output vocabulary projection, bias, normalization, activations, softmax and cache preparation; it does not model causal-kernel savings. It is not total training/inference FLOPs or measured latency. Parameter-cap selection uses the archive's reported **total** model parameters, which include more than the NSC-scored matrices. Report this distinction.

## Evaluation protocol

1. **Mechanism check:** construct a small counterfactual such as the reorder pair above, check invariants and limiting cases, and show that the ablation removes the proposed effect. This is separate from the predictive claim.
2. **Ranking:** use development validation PPL, then final test PPL on different architectures. Correlate scores with **negative** PPL. The evaluator reports average-rank Spearman rho and Kendall tau-b, including score ties. Constant-score correlation is `null`, not zero. Scores are rounded to 12 significant digits only for comparisons.
3. **Size control:** alongside all pairs, report pairs with the same hidden width and parameter ratio min/max at least 0.90. The pair statistic uses the tau-b formula on that restricted set; it is not a standard full-panel rank correlation or a causal control. Report its pair count (180 on the development split). These pairs overlap; do not use pair count as an independent sample size. The baseline does not match FLOPs as well as parameters.
4. **Selection:** use parameter caps of 10M, 20M, 30M and 50M, and k=1 and k=3. Each method chooses its k highest scores within the same feasible pool before looking at outcomes. Reveal the recorded PPL of those k designs and report the best. Regret is that best PPL minus the minimum PPL anywhere in the feasible pool. Compare original NSC, the positional example, your method, its ablation, reported parameters, reference FLOPs and uniform random selection. This is simulated selection using archived outcomes, not a new online training/search run.
5. **Randomness:** 200 fixed-seed trials randomize score ties or random selections. The standard deviation is over those choices; it is not training-seed uncertainty. A deterministic selection has zero such variation. Bootstrap confidence intervals are optional; if added, resample architectures, not individual pairs.

All methods use the same k and resource caps. The baseline caps and k values must remain in the final report; additional development-chosen analyses are welcome. A good report can show no improvement, expose failures of the supplied positional score, or explain why a structural distinction does not help ranking. Describe what your score learns from the graph beyond model size. Finite-pool top-k selection is sufficient; a new search algorithm is optional.

## Provenance and scope of the evidence

The course benchmark archive supplies this 200-configuration panel and attributes it to [LiteTransformerSearch, NeurIPS 2022](https://arxiv.org/abs/2203.02094). It was used in the [NSC](https://arxiv.org/abs/2609.23087) GPT-2 benchmark experiment. The archive uses the names GPT-2 and Transformer-XL; the preserved configurations include adaptive embeddings, memory length 192 and `pre_lnorm=false`. Describe the actual archived configuration family, not an assumed modern pre-normalized GPT architecture.

The source JSON contains one recorded outcome per architecture, but not training seeds, steps/tokens, hardware or the exact upstream training revision. We have not independently retrained the panel or reconstructed that provenance. Results support an exploratory comparison on this fixed archive. They cannot establish training-seed significance, full-pretraining performance, or generalization to another architecture family. The final split is an architecture holdout within this archive, not an independent benchmark.

The panel varies width, depth, head counts and FFN allocations. It does **not** contain controlled trained reorder pairs, MoE, GQA/MLA, tied loops or hyperconnections. Synthetic scores for these extensions cannot be assigned an existing architecture's PPL. Such directions need corresponding trained outcomes or a clearly bounded model-validation study. Training is optional and outside the supplied harness. Cite the published methods you build on and distinguish reproduction from your own contribution.

## References

- Zhu, Zhao and Lu. *Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone.* [Paper](https://arxiv.org/abs/2609.23087); [official implementation](https://github.com/Optima-CityU/neural-spectral-capacity).
- Javaheripi et al. *LiteTransformerSearch: Training-free Neural Architecture Search for Efficient Language Models.* [Paper](https://arxiv.org/abs/2203.02094); [author code](https://github.com/microsoft/archai/tree/neurips-lts/archai/nlp).

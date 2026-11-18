# Where Should Capacity Count? Residual Transport and the Embedding Path in NSC

CS5491 Artificial Intelligence course project (CityU, 2026/27 Semester A).
Team: Kaiwen Wang, Leao Wang, Lei Gao.

## Task

[Neural Spectral Capacity (NSC)](https://arxiv.org/abs/2609.23087) scores a neural architecture from its
specification alone: it sums the Marchenko–Pastur capacity `psi_MP` of every attention and FFN weight
matrix, with **equal weight across layers**, and **ignores the embedding**. The course topic asks for one
motivated change that makes NSC sensitive to a computational-graph mechanism, and an evaluation of whether it
improves ranking and budgeted selection of 200 archived, trained decoder Transformers (LiteTransformerSearch
panel, perplexity outcomes).

**Study 1 (v2, tag `v2-frozen`): residual transport weights.** We derive, from the specification only, how
strongly each sublayer's output is transported to the network output at initialization (`T_s`), for Post-LN,
Pre-LN and unnormalized residual networks, with a mean-field recursion that accounts for token correlation
and Transformer-XL's detached memory. Score `S = sum_s T_s * Psi_s`; ablation `T_s = 1` = original NSC.
- The prediction matches Monte Carlo on random residual networks (0.8% RMS error, Post-LN).
- In Post-LN, which every archived model uses, LayerNorm divides out residual growth, so `T_s` is almost
  uniform (0.84–1.00). Final-split Spearman rho 0.9667 vs NSC 0.9659: a supported negative result.

**Study 2 (v3, tag `v3-frozen`): capacity of the embedding path.** Every token enters through a tied
adaptive-embedding table (v tokens x w columns) and a w x d_model projection. Using Study 1's result that
Post-LN removes the projection's scale, the path's capacity is `psi_MP(w, d, 0.02*sqrt(v/w))` per table.
Score = NSC + this term; ablation = NSC.
- In the archive, embedding width explains much of what NSC misses within each model width. Study 3
  below tests the predicted saturation near `d_embed = d_model` using controlled training.
- The formula matches Monte Carlo spectra within 0.4% RMS.
- Final split (pre-registered, run once): rho **0.9802** vs NSC 0.9659, bootstrap delta +0.015
  [+0.006, +0.028]; within d=512, 0.749 vs 0.659. Budgeted selection is unchanged.
- Disclosure: the final split was used twice (v2, then v3). v3 was designed on development data only.

**Study 3 (v4): controlled training test of the embedding term.** We trained 47 Transformer-XL models
on WikiText-103 with GPT-2 BPE, keeping the decoder fixed within each panel and varying `d_embed` and
`div_val`. The configurations, hypotheses and analysis were pre-registered.
- Larger embeddings improve validation perplexity. In panel A, v3 ranks configurations with rho 0.825
  (one-sided permutation p = 0.001), while total parameter count reaches rho 0.993.
- Two structural predictions fail: the improvement continues beyond `d_embed = d_model`, and
  `div_val = 2` gives worse perplexity than `div_val = 1` at every tested embedding width.
- The embedding effect persists at three times the training budget (log-PPL gap ratio 0.95).
  These results support including embeddings in the score, while limiting the proposed spectral
  mechanism's interpretation. The test covers one dataset and model family with short training.

Report: [`report/report.pdf`](report/report.pdf) (source `report/report.tex`), including all three studies.

## Repository layout

| Path | Contents |
|---|---|
| `nsc-starter/` | Course starter: NSC scorer (`nsc.py`), evaluator (`evaluate.py`), self-check, data |
| `nsc-starter/residual_transport.py` | **Study 1**: specification-only transport factors `T_s` (standard library only) |
| `nsc-starter/student_score.py` | **Study 1**: `score` = transport-weighted NSC (Post-LN); `ablation` = original NSC |
| `nsc-starter/embedding_path.py` | **Study 2**: embedding path capacity (standard library only) |
| `nsc-starter/v3_score.py` | **Study 2**: `score` = NSC + embedding path capacity; `ablation` = original NSC |
| `analysis/v3_embedding_path/` | Study 2: data analysis, mechanism checks, every development variant examined, final results |
| `analysis/meanfield_transport/` | NumPy reference of the recursion, Monte Carlo validation, mechanism checks |
| `analysis/postln_transport/` | Earlier i.i.d.-token version (v1), kept as a documented baseline of the derivation |
| `analysis/residual_foundations/` | Small controlled checks on residual learning (motivation, scope) |
| `analysis/starter_audit/` | Reproduction of the unmodified starter |
| `analysis/dev_eval/` | Extended evaluation with all baselines/controls (`--split development` or `final`) |
| `analysis/final_eval/` | One-shot final-split results |
| `controlled/` | Study 3: data preparation, Transformer-XL training, pre-registered panel and analysis |
| `controlled/runs/` | Results of all 47 panel runs and 3 timing pilots |
| `controlled/README.md` | Study 3 setup, pre-registration, execution record and interpretation |
| `report/` | Report source, figure script and PDF (public build lists names only) |

The notes in `analysis/*/README.md` are in Chinese. They record the assumptions, intermediate results and
decisions behind each step.

## Environment

- The scorer and evaluator (`nsc-starter/`) need **Python 3.9+, standard library only**.
- The analysis scripts need NumPy and Matplotlib. Tested with Python 3.12.14, numpy 2.5.4 and
  matplotlib 3.11.2 on Windows 11.
- Building the report needs XeLaTeX (TeX Live 2026) and the Times New Roman font.
- Study 3 training additionally needs the packages in `controlled/requirements.txt` and a CUDA GPU.
  The recorded runs used PyTorch 2.9.1 with CUDA 12.8; analysis of the saved results needs only NumPy.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Every command below runs from the repository root. CPU analysis times were measured on an Intel Core Ultra 9
290HX (24 threads). Study 3 GPU timings are listed separately below.

## Reproducing every result

### 1. Starter self-check and evaluation (seconds)

```sh
python nsc-starter/check_starter.py
python nsc-starter/evaluate.py --split development --output analysis/dev_eval/development_results.json
```

Expected output: `PASS: MP identities/convergence, ...`. The development evaluation prints:

```
original_nsc         rho=  0.969745 tau=  0.861414 controlled_tau=  0.622222
position_example     rho=  0.970069 tau=  0.865051 controlled_tau=  0.622222
student              rho=  0.969913 tau=  0.861414 controlled_tau=  0.622222
ablation             rho=  0.969745 tau=  0.861414 controlled_tau=  0.622222
reported_parameters  rho=  0.727873 tau=  0.575758 controlled_tau=  0.133333
reference_flops      rho=  0.967948 tau=  0.858471 controlled_tau=  0.622222
```

### 2. Extended baselines and controls (about 5 s per split; Study 1 rows of Tables 1–2)

```sh
python analysis/dev_eval/extended_eval.py --split development
python analysis/dev_eval/extended_eval.py --split final
```

These write `analysis/dev_eval/{development,final}_extended.json` and print ranking, ties, regret, best
selected PPL and scoring time (Section 4 of the report; scoring time in Section 5).

### 3. Study 1 final split (run once at `v2-frozen`; seconds)

```sh
python nsc-starter/evaluate.py --split final --output analysis/final_eval/final_results.json
```

Expected `student rho=0.966661`, `ablation rho=0.965941`. The method was frozen at git tag `v2-frozen`
before this command was first run. See `analysis/final_eval/README.md`.

### 4. Study 1 derivation checks (no labels read)

| Command | Time | Output | Report |
|---|---|---|---|
| `python analysis/meanfield_transport/strengthen.py` | ~27 min (22 processes) | `strengthen.json`, `strengthen_rows.json` | Section 2, Fig. 1a |
| `python analysis/meanfield_transport/mechanism_checks.py` | ~1 min | `mechanism_checks.json` (reorder pair, panel weights, strong-signal regime) | Section 2 (panel weights) |
| `python analysis/meanfield_transport/meanfield.py` | ~12 min | `validation.json` (first validation at n=64; v1 vs v2) | notes only (v1 vs v2) |
| `python analysis/postln_transport/transport.py` | ~3.5 min | v1 results | notes only |
| `python analysis/residual_foundations/verify.py` | ~2 s | zero-output-branch counterexample and related checks | notes only (scope) |

Monte Carlo seeds are fixed, so reruns reproduce the stored JSON up to floating-point noise.
`strengthen.py` uses all but two CPU cores; with fewer cores it runs proportionally longer.

### 5. Study 2 (v3): embedding path capacity

Evaluation (seconds each; Tables 1–2 of the report):

```sh
python nsc-starter/evaluate.py --split development --student nsc-starter/v3_score.py
python analysis/v3_embedding_path/v3_final_eval.py --split development
python nsc-starter/evaluate.py --split final --student nsc-starter/v3_score.py --output analysis/v3_embedding_path/v3_final_results.json
python analysis/v3_embedding_path/v3_final_eval.py --split final
```

Expected: development `student rho= 0.983174`; final `student rho= 0.980246`, `ablation rho= 0.965941`, and
`bootstrap rho(v3) - rho(NSC): +0.0151  95% CI [+0.0064, +0.0281]`. The final commands were run once, at tag
`v3-frozen`, following the plan in `analysis/v3_embedding_path/README.md` section 10.

Analysis behind the method (development labels only, except the label-free mechanism check):

| Command | Time | What it shows | Report |
|---|---|---|---|
| `python analysis/v3_embedding_path/explore.py` | ~2 s | residual of log PPL vs structural features, by width | Section 3 |
| `python analysis/v3_embedding_path/mechanism_check.py` | ~50 s | label-free: Monte Carlo path spectra vs formula, controlled pair | Section 3 |
| `python analysis/v3_embedding_path/embedding_capacity.py` | ~2 s | uncapped embedding (ablation of the bottleneck) and early variants | Section 4 |
| `python analysis/v3_embedding_path/dev_eval_path.py` | ~2 s | development evaluation of the chosen formula, bootstrap | Section 4 |
| `python analysis/v3_embedding_path/failures.py` | ~2 s | where NSC and v3 still fail | Section 5 |

The notes in `analysis/v3_embedding_path/README.md` record every variant examined on development labels and
the series-aggregation test (pre-registered, failed), in order.

### 6. Study 3 (v4): controlled training and analysis

Analyze the committed results without retraining (seconds):

```sh
python controlled/analyze.py
```

This writes `controlled/analysis.json`. Expected results include panel A rho(v3) = 0.825,
rho(total parameters) = 0.993, pooled seed standard deviation = 0.0150, H1 and H4 passing,
H2 and H3 failing, and training-length gap ratio = 0.95. The report's Study 3 table uses these results.
See [`controlled/README.md`](controlled/README.md) for the full pre-registration and results.

To reproduce training, install the CUDA build of PyTorch and the remaining dependencies, then prepare data:

```sh
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r controlled/requirements.txt
python controlled/prepare_data.py
```

Preparation downloads SHA-256-verified WikiText-103 files and tokenizes them with GPT-2 BPE, producing
`controlled/data/{train,valid,test}.npy` (about 0.5 GB). Expected token counts are 117,920,140 / 247,289 /
283,287. Download time depends on the network; tokenization takes a few minutes. Outcomes use validation
perplexity; the test split is unused.

`run_panel.py` skips any run whose JSON already exists. To retrain the full panel, first move the 47
`A_*.json`, `B_*.json` and `C_*.json` files from `controlled/runs/` to a backup directory, then run:

```sh
python controlled/run_panel.py --dry
python controlled/run_panel.py
python controlled/analyze.py
```

The dry run should list 47 pending runs (36 in A, 9 in B, 2 in C). The recorded panel took 6.3 GPU-hours
on an RTX 4080 SUPER: A runs took 326–513 s each, B 476–589 s and C 1019–1248 s. Two concurrent workers
using `--shard 0/2` and `--shard 1/2` completed it in about 3.3 hours. Seeded GPU reruns may differ
numerically across hardware and software environments.

### 7. Report figure and PDF

```sh
python report/make_figure.py       # writes report/figure1.pdf (panel b reads development labels)
cd report && xelatex report.tex && xelatex report.tex
```

## Data and provenance

`nsc-starter/data/` is the course-supplied archive. `provenance.json` describes its sources and SHA-256
hashes, which `check_starter.py` verifies. Git stores these files byte-exact (see `.gitattributes`). Outcome
labels are read by evaluation and outcome-analysis scripts. The scorer receives the configuration only.
Study 3 uses independently trained outcomes and does not use the archive's final-split labels.

## Use of LLMs

Claude (Anthropic) and ChatGPT/Codex (OpenAI) assisted with literature reading, code, analysis scripts and
drafting. No LLM is called by any experiment.

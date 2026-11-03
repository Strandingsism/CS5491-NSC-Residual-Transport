# Where Should Layers Count? Residual Transport Weighting for NSC

CS5491 Artificial Intelligence course project (CityU, 2026/27 Semester A).
Team: Kaiwen Wang, Leao Wang, Lei Gao.

## Task

[Neural Spectral Capacity (NSC)](https://arxiv.org/abs/2609.23087) scores a neural architecture from its
specification alone: it sums the Marchenko–Pastur capacity `psi_MP` of every weight matrix, with **equal
weight across layers**. The course topic asks for one motivated change that makes NSC sensitive to a
computational-graph mechanism, and an evaluation of whether it improves ranking and budgeted selection of
200 archived, trained decoder Transformers (LiteTransformerSearch panel, perplexity outcomes).

**Our change.** We derive, from the specification only, how strongly each sublayer's output is transported
to the network output at initialization (`T_s`), for Post-LN, Pre-LN and unnormalized residual networks. The
derivation is a mean-field recursion that accounts for token correlation and Transformer-XL's detached
memory. The score is `S = sum_s T_s * Psi_s`. The ablation sets `T_s = 1`, which recovers original NSC
exactly.

**Findings.**
- The prediction matches Monte Carlo measurements on random residual networks (0.8% RMS error in the
  archive's Post-LN setting).
- In Post-LN, which every archived model uses, residual growth and LayerNorm nearly cancel, so `T_s` is
  almost uniform (0.84–1.00). Pre-LN and no normalization give strongly position-dependent weights.
- Ranking and selection are therefore essentially unchanged. On the held-out final split, Spearman
  rho = 0.9667 for our score versus 0.9659 for NSC, and top-k selection is identical. This is a supported
  negative result.

Report: [`report/report.pdf`](report/report.pdf) (source `report/report.tex`).

## Repository layout

| Path | Contents |
|---|---|
| `nsc-starter/` | Course starter: NSC scorer (`nsc.py`), evaluator (`evaluate.py`), self-check, data |
| `nsc-starter/residual_transport.py` | **Our method**: specification-only transport factors `T_s` (standard library only) |
| `nsc-starter/student_score.py` | **Our method**: `score` = transport-weighted NSC (Post-LN); `ablation` = original NSC |
| `analysis/meanfield_transport/` | NumPy reference of the recursion, Monte Carlo validation, mechanism checks |
| `analysis/postln_transport/` | Earlier i.i.d.-token version (v1), kept as a documented baseline of the derivation |
| `analysis/residual_foundations/` | Small controlled checks on residual learning (motivation, scope) |
| `analysis/starter_audit/` | Reproduction of the unmodified starter |
| `analysis/dev_eval/` | Extended evaluation with all baselines/controls (`--split development` or `final`) |
| `analysis/final_eval/` | One-shot final-split results |
| `report/` | Report source, figure script and PDF (public build lists names only) |

The notes in `analysis/*/README.md` are in Chinese. They record the assumptions, intermediate results and
decisions behind each step.

## Environment

- The scorer and evaluator (`nsc-starter/`) need **Python 3.9+, standard library only**.
- The analysis scripts need NumPy and Matplotlib. Tested with Python 3.12.14, numpy 2.5.4 and
  matplotlib 3.11.2 on Windows 11.
- Building the report needs XeLaTeX (TeX Live 2026) and the Times New Roman font.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Every command below runs from the repository root. Reported times were measured on an Intel Core Ultra 9
290HX (24 threads).

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

### 2. Extended baselines and controls (about 5 s per split; Tables 2–3 of the report)

```sh
python analysis/dev_eval/extended_eval.py --split development
python analysis/dev_eval/extended_eval.py --split final
```

These write `analysis/dev_eval/{development,final}_extended.json` and print ranking, ties, regret, best
selected PPL and scoring time (Section 4 of the report).

### 3. Final split (run once, after freezing; seconds)

```sh
python nsc-starter/evaluate.py --split final --output analysis/final_eval/final_results.json
```

Expected `student rho=0.966661`, `ablation rho=0.965941`. The method was frozen at git tag `v2-frozen`
before this command was first run. See `analysis/final_eval/README.md`.

### 4. Derivation checks (no labels read)

| Command | Time | Output | Report |
|---|---|---|---|
| `python analysis/meanfield_transport/strengthen.py` | ~27 min (22 processes) | `strengthen.json`, `strengthen_rows.json` | Table 1, Fig. 1a |
| `python analysis/meanfield_transport/mechanism_checks.py` | ~1 min | `mechanism_checks.json` (reorder pair, panel weights, strong-signal regime) | Section 3, Fig. 1b |
| `python analysis/meanfield_transport/meanfield.py` | ~12 min | `validation.json` (first validation at n=64; v1 vs v2) | Section 3 (v1 error) |
| `python analysis/postln_transport/transport.py` | ~3.5 min | v1 results | Section 3 (v1 error) |
| `python analysis/residual_foundations/verify.py` | ~2 s | zero-output-branch counterexample and related checks | Section 5 |

Monte Carlo seeds are fixed, so reruns reproduce the stored JSON up to floating-point noise.
`strengthen.py` uses all but two CPU cores; with fewer cores it runs proportionally longer.

### 5. Report figure and PDF

```sh
python report/make_figure.py       # writes report/transport_profiles.pdf from the MC outputs
cd report && xelatex report.tex && xelatex report.tex
```

## Data and provenance

`nsc-starter/data/` is the course-supplied archive. `provenance.json` describes its sources and SHA-256
hashes, which `check_starter.py` verifies. Git stores these files byte-exact (see `.gitattributes`). Outcome
labels are read only by `evaluate.py` and `extended_eval.py`. The scorer receives the configuration only.

## Use of LLMs

Claude (Anthropic) and ChatGPT/Codex (OpenAI) assisted with literature reading, code, analysis scripts and
drafting. No LLM is called by any experiment.

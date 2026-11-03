"""Strengthened Monte Carlo validation of the v2 transport prediction (pre-final, label-free).

Addresses the limitations listed in README.md for validation.json:
- archive sequence setting n = mem = 192 (was 64);
- 10 seeds x 2 probes per (config, placement), keeping per-seed measurements so that
  Monte Carlo standard errors can be reported and compared with the prediction error;
- 28 development specifications: per hidden width, depth quantiles 0, .2, .4, .5, .6, .8, 1
  (superset of the original 12-config subset);
- bootstrap over configurations for the error summaries.

The reference model and the prediction are unchanged (meanfield.py / residual_transport.py).
No label file is read. Runs (key, placement, seed) tasks in parallel, one BLAS thread each.

Run from project root: .venv/Scripts/python.exe analysis/meanfield_transport/strengthen.py
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import meanfield as mf  # noqa: E402  (also puts nsc-starter on sys.path)
from nsc import layers  # noqa: E402
from residual_transport import transport_factors  # noqa: E402

N = MEM = 192
SEEDS = 10
PROBES = 2
QUANTILES = (0, 0.2, 0.4, 0.5, 0.6, 0.8, 1)
BOOT = 2000


def subset(configs, keys):
    by_width = {}
    for k in keys:
        by_width.setdefault(configs[k]['d_model'], []).append(k)
    out = []
    for w in sorted(by_width):
        ks = sorted(by_width[w], key=lambda k: configs[k]['n_layer'])
        idx = sorted({min(len(ks) - 1, int(q * len(ks))) for q in QUANTILES} | {len(ks) // 2})
        out += [ks[i] for i in idx]
    return out


def task(args):
    key, config, norm, seed = args
    d, heads, inner = layers(config)
    # Seeds 1000+ are disjoint from validation.json (seeds 0, 1).
    T, R = measure_one(d, heads, inner, norm, 1000 + seed)
    return key, norm, seed, T.tolist(), R.tolist()


def measure_one(d, heads, inner, norm, rng_seed, eps=1e-4, warmup=2):
    """meanfield.measure for a single explicit seed (same procedure)."""
    S = 2 * len(inner)
    T, R = np.zeros(S), np.zeros(S)
    rng = np.random.default_rng(rng_seed)
    subs = mf.make_sublayers(d, heads, inner, rng)
    mems = None
    for _ in range(warmup):
        rec = []
        mf.run(rng.normal(size=(MEM, d)), subs, norm, mems, record=rec)
        mems = rec
    x0 = rng.normal(size=(N, d))
    rec = []
    mf.run(x0, subs, norm, mems, record=rec)
    for s in range(S):
        R[s] = mf.mean_pairwise_cos(rec[s])
        for _ in range(PROBES):
            v = rng.normal(size=(N, d))
            diff = mf.run(rec[s], subs, norm, mems, s, eps * v) - mf.run(rec[s], subs, norm, mems, s, -eps * v)
            T[s] += (diff * diff).sum() / (4 * eps * eps) / (N * d) / PROBES
    return T, R


def boot_ci(per_config, stat, rng):
    """Percentile CI of stat(concatenated arrays) resampling configurations."""
    n = len(per_config)
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, n, n)
        vals.append(stat(np.concatenate([per_config[i] for i in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(rows):
    rng = np.random.default_rng(5491)
    out = {}
    for norm in mf.NORMS:
        sel = [r for r in rows if r['norm'] == norm]
        rel, z, rel_tail, cfg_rel, prof = [], [], [], [], []
        for r in sel:
            M = np.array(r['measured_seeds'])           # seeds x sublayers
            mean, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
            pred = np.array(r['v2'])
            rel.append(pred / mean - 1)
            rel_tail.append((pred / mean - 1)[1:])       # excludes the first sublayer (input artifact)
            z.append((pred - mean) / se)
            cfg_rel.append(np.array([pred.mean() / mean.mean() - 1]))
            if len(pred) > 4 and np.std(pred) > 0:
                prof.append(float(np.corrcoef(pred, mean)[0, 1]))
            r['measured_se'] = se.tolist()
        rms = lambda a: float(np.sqrt((a ** 2).mean()))
        allrel, allz = np.concatenate(rel), np.concatenate(z)
        out[norm] = dict(
            configs=len(sel), sublayers=int(allrel.size),
            rel_error_mean=float(allrel.mean()), rel_error_rms=rms(allrel),
            rel_error_rms_ci95=boot_ci(rel, rms, rng),
            rel_error_max_abs=float(np.abs(allrel).max()),
            rel_error_rms_excl_first=rms(np.concatenate(rel_tail)),
            config_mean_T_rel_error_rms=rms(np.concatenate(cfg_rel)),
            config_mean_T_rel_error_max_abs=float(np.abs(np.concatenate(cfg_rel)).max()),
            mc_rel_se_median=float(np.median(np.concatenate(
                [np.array(r['measured_se']) / np.array(r['measured_seeds']).mean(0) for r in sel]))),
            z_rms=rms(allz), frac_within_2se=float((np.abs(allz) < 2).mean()),
            profile_corr_mean=float(np.mean(prof)) if prof else None,
            profile_corr_median=float(np.median(prof)) if prof else None,
            measured_max_over_min_median=float(np.median(
                [max(np.mean(r['measured_seeds'], 0)) / min(np.mean(r['measured_seeds'], 0)) for r in sel])),
            rho_last_measured_mean=float(np.mean([np.mean(r['measured_rho_seeds'], 0)[-1] for r in sel])),
            rho_last_v2_mean=float(np.mean([r['v2_rho'][-1] for r in sel])))
    return out


def main():
    start = time.perf_counter()
    data = ROOT / 'nsc-starter' / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    keys = json.loads((data / 'splits.json').read_text())['development']
    chosen = subset(configs, keys)
    tasks = [(k, configs[k], norm, s) for k in chosen for norm in mf.NORMS for s in range(SEEDS)]
    # longest tasks first for better packing
    tasks.sort(key=lambda t: -(t[1]['d_model'] ** 2 * t[1]['n_layer'] ** 2))
    workers = max(1, (os.cpu_count() or 2) - 2)
    results = {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (key, norm, seed, T, R) in enumerate(pool.map(task, tasks, chunksize=1), 1):
            results.setdefault((key, norm), {})[seed] = (T, R)
            if i % 50 == 0:
                print(f'{i}/{len(tasks)} {time.perf_counter() - start:.0f}s', flush=True)
    rows = []
    for key in chosen:
        d, heads, inner = layers(configs[key])
        for norm in mf.NORMS:
            per = results[(key, norm)]
            rows.append(dict(key=key, d=d, L=len(inner), norm=norm, n=N, mem=MEM,
                             measured_seeds=[per[s][0] for s in range(SEEDS)],
                             measured_rho_seeds=[per[s][1] for s in range(SEEDS)],
                             v2=transport_factors(d, inner, norm, N, MEM),
                             v2_rho=mf.predicted_correlation(d, inner, norm, N, MEM)))
    summary = summarize(rows)
    result = dict(setting=dict(n=N, mem=MEM, seeds=SEEDS, probes=PROBES, quantiles=QUANTILES,
                               bootstrap=BOOT),
                  subset=chosen, summary=summary,
                  hardware=dict(processor=platform.processor(), cpus=os.cpu_count(), workers=workers,
                                python=platform.python_version(), numpy=np.__version__),
                  seconds=time.perf_counter() - start)
    (OUT / 'strengthen.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (OUT / 'strengthen_rows.json').write_text(json.dumps(rows) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1))
    print(f"{result['seconds']:.0f}s")


if __name__ == '__main__':
    main()

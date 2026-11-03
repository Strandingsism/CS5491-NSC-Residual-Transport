"""Pre-registered analysis of the controlled panel (README section 3); written before any panel run.

Reads controlled/runs/{A,B,C}_*.json. Outcome: validation log PPL (nats/token), averaged over seeds.
Scores are computed from the specification with the frozen v3 code (nsc-starter/v3_score.py) and
compared with original NSC and total parameter count.

Run from project root: .venv/Scripts/python.exe controlled/analyze.py   (writes controlled/analysis.json)
"""
from pathlib import Path
import itertools
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'nsc-starter'))
from nsc import original_nsc  # noqa: E402
import v3_score  # noqa: E402

N_TOKEN, CUTOFFS = 50257, [19997, 39997, 50257]


def archive_config(cfg):
    """Archive-style dict for the frozen scorers."""
    L = len(cfg['n_head'])
    return dict(n_token=N_TOKEN, cutoffs=CUTOFFS, d_model=cfg['d_model'], n_layer=L, n_head=cfg['n_head'],
                d_inner=cfg['d_inner'], d_embed=cfg['d_embed'], div_val=cfg['div_val'], d_head=-1,
                mem_len=cfg['mem_len'], tgt_len=cfg['mem_len'])


def rankdata(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    order = np.argsort(x, kind='mergesort')
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float('nan') if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])


def perm_p(score, quality, n=20000, seed=5491):
    """One-sided permutation p-value of Spearman rho(score, quality) > observed under random labels."""
    rng = np.random.default_rng(seed)
    obs = spearman(score, quality)
    q = np.asarray(quality)
    return float(np.mean([spearman(score, rng.permutation(q)) >= obs for _ in range(n)]))


def load(prefix):
    """Group runs by configuration (name without seed): config, params and per-seed log PPL."""
    groups = {}
    for f in sorted((HERE / 'runs').glob(f'{prefix}_*.json')):
        r = json.loads(f.read_text())
        key = r['name'].rsplit('_s', 1)[0]
        g = groups.setdefault(key, {'config': r['config'], 'params': r['params'], 'nll': []})
        g['nll'].append(r['valid_nll'])
    for g in groups.values():
        c = archive_config(g['config'])
        g['nsc'], g['v3'] = original_nsc(c), v3_score.score(c)
        g['mean'] = float(np.mean(g['nll']))
        g['se'] = float(np.std(g['nll'], ddof=1) / math.sqrt(len(g['nll']))) if len(g['nll']) > 1 else float('nan')
    return groups


def diff(groups, a, b):
    """mean log PPL(a) - mean log PPL(b) and its standard error."""
    ga, gb = groups[a], groups[b]
    return ga['mean'] - gb['mean'], math.hypot(ga['se'], gb['se'])


def main():
    out = {}
    A, B, C = load('A'), load('B'), load('C')
    # seed noise: pooled within-configuration SD of log PPL (panels A and B)
    resid = [x - np.mean(g['nll']) for g in itertools.chain(A.values(), B.values()) if len(g['nll']) > 1 for x in g['nll']]
    dof = sum(len(g['nll']) - 1 for g in itertools.chain(A.values(), B.values()) if len(g['nll']) > 1)
    out['seed_sd_log_ppl'] = float(math.sqrt(sum(r * r for r in resid) / dof)) if dof else float('nan')
    print(f'seed noise: pooled SD of log PPL = {out["seed_sd_log_ppl"]:.4f}  (archive upper bound ~0.02)')

    for name, G in (('A', A), ('B', B)):
        keys = sorted(G)
        q = [-G[k]['mean'] for k in keys]
        res = {}
        for s in ('v3', 'nsc'):
            v = [G[k][s] for k in keys]
            res[s] = {'rho': spearman(v, q), 'p': perm_p(v, q) if len(set(v)) > 1 else None}
        tot = [G[k]['params']['total'] for k in keys]
        res['total_params'] = {'rho': spearman(tot, q), 'p': perm_p(tot, q)}
        res['configs'] = {k: {'mean_log_ppl': G[k]['mean'], 'se': G[k]['se'], 'seeds': len(G[k]['nll']),
                              'v3': G[k]['v3'], 'nsc': G[k]['nsc'], 'total_params': G[k]['params']['total']} for k in keys}
        out[f'panel_{name}'] = res
        print(f'\npanel {name}: {len(keys)} configurations')
        for k in keys:
            g = G[k]
            print(f'  {k:10s} log PPL {g["mean"]:.4f} +- {g["se"]:.4f} (n={len(g["nll"])})  PPL {math.exp(g["mean"]):7.2f}  '
                  f'v3 {g["v3"]:8.1f}  params {g["params"]["total"] / 1e6:6.2f}M')
        for s in ('v3', 'nsc', 'total_params'):
            print(f'  rho({s}, -log PPL) = {res[s]["rho"]:+.3f}  p = {res[s]["p"]}')

    tests = {}
    if {'A_e256_v1', 'A_e512_v1'} <= set(A):  # H2: saturation at d_embed = d_model
        d, se = diff(A, 'A_e512_v1', 'A_e256_v1')
        tests['H2_saturation'] = {'diff_512_minus_256': d, 'se': se, 'holds': d >= -2 * se}
    h3 = {}
    for de in (64, 128, 256, 512):  # H3: div_val 2 beats div_val 1 at the same d_embed
        if {f'A_e{de}_v1', f'A_e{de}_v2'} <= set(A):
            d, se = diff(A, f'A_e{de}_v2', f'A_e{de}_v1')
            h3[str(de)] = {'diff_v2_minus_v1': d, 'se': se, 'v2_better_2se': d < -2 * se, 'v2_worse_2se': d > 2 * se}
    if h3:
        tests['H3_div2_vs_div1'] = {'by_d_embed': h3, 'holds': sum(v['v2_better_2se'] for v in h3.values()) >= 3}
    if all(f'B_e{e}_v1' in B for e in (128, 256, 512)):  # H4: monotone in d_embed at d_model 512
        m = [B[f'B_e{e}_v1']['mean'] for e in (128, 256, 512)]
        tests['H4_wide_monotone'] = {'means': m, 'holds': m[0] > m[1] > m[2]}
    if {'C_e64_v1', 'C_e256_v1'} <= set(C) and {'A_e64_v1', 'A_e256_v1'} <= set(A):
        short = A['A_e64_v1']['mean'] - A['A_e256_v1']['mean']
        long = C['C_e64_v1']['mean'] - C['C_e256_v1']['mean']
        tests['C_length'] = {'gap_2e7': short, 'gap_6e7': long, 'ratio': long / short, 'serious_caveat': long / short < 0.5}
    out['tests'] = tests
    print('\npre-registered tests:')
    for k, v in tests.items():
        print(f'  {k}: {json.dumps(v)}')
    (HERE / 'analysis.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

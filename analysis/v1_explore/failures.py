"""Where do NSC and v3 both fail? (development split only)

Residual of log PPL after a common-slope fit on log score with width intercepts, for NSC and v3
(NSC + embedding path capacity). Correlates residuals with structural features within each width
(stratified Spearman), lists worst outliers, and inspects discordant size-controlled pairs.
Positive residual = model is worse than its score predicts (score over-rates it).
Run from project root: .venv/Scripts/python.exe analysis/v1_explore/failures.py
"""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsc import layers, original_nsc, resource_estimate  # noqa: E402
from embedding_capacity import nsc_path, path_embedding_capacity  # noqa: E402

D = ROOT / 'nsc-starter' / 'data'
configs = json.loads((D / 'architectures.json').read_text())
res = json.loads((D / 'resources.json').read_text())
keys = json.loads((D / 'splits.json').read_text())['development']
lab = json.loads((D / 'development_labels.json').read_text())


def rk(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


feat = {}
for k in keys:
    c = configs[k]
    d, heads, inner = layers(c)
    L = len(inner)
    proj = resource_estimate(c)['projection_parameters']
    feat[k] = dict(
        L=L, aspect=d / L, heads=np.mean(heads), head_dim=np.mean([d / h for h in heads]),
        heads_cv=np.std(heads) / np.mean(heads), ffn_ratio=np.mean(inner) / d, ffn_cv=np.std(inner) / np.mean(inner),
        ffn_min=min(inner) / d, ffn_max=max(inner) / d,
        ffn_trend=spearman(range(L), inner) if L > 2 else 0.0,
        attn_frac=4 * d * d * L / proj, d_embed=c['d_embed'], div_val=c['div_val'],
        emb_over_nsc=path_embedding_capacity(c) / original_nsc(c),
        log_proj=math.log(proj))

W = np.array([configs[k]['d_model'] for k in keys])
y = np.log([lab[k]['valid_ppl'] for k in keys])
widths = (64, 128, 256, 512)


def residual(score):
    X = np.column_stack([(W == w).astype(float) for w in widths] + [np.log(score)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta, beta[-1]


S = {'NSC': np.array([original_nsc(configs[k]) for k in keys]), 'v3': np.array([nsc_path(configs[k]) for k in keys])}
R = {}
for n, s in S.items():
    R[n], slope = residual(s)
    print(f'{n}: slope {slope:+.3f}, residual sd {R[n].std():.4f}; within-width sd ' +
          ' '.join(f'{R[n][W == w].std():.4f}' for w in widths))
print(f'corr(resid NSC, resid v3) = {np.corrcoef(R["NSC"], R["v3"])[0, 1]:.3f}')

print('\nStratified Spearman(feature, v3 residual) by width   [+ = feature makes model worse than predicted]')
names = list(next(iter(feat.values())))
print(f"{'feature':14s} " + ' '.join(f'd{w:<5d}' for w in widths) + '  pooled(within-width ranks)')
for f in names:
    vals = np.array([feat[k][f] for k in keys])
    per = [spearman(vals[W == w], R['v3'][W == w]) for w in widths]
    # pooled: rank-transform within width, then correlate
    vr, rr = np.zeros(100), np.zeros(100)
    for w in widths:
        m = W == w
        vr[m] = (rk(vals[m]) + 0.5) / m.sum()
        rr[m] = (rk(R['v3'][m]) + 0.5) / m.sum()
    pooled = float(np.corrcoef(vr, rr)[0, 1]) if vr.std() > 0 else float('nan')
    print(f'{f:14s} ' + ' '.join(f'{v:+.2f} ' for v in per) + f'  {pooled:+.3f}')

print('\nWorst v3 outliers (|residual|), with NSC residual')
for i in np.argsort(-np.abs(R['v3']))[:12]:
    k = keys[i]
    fk = feat[k]
    print(f"  {k:13s} d={W[i]:3d} L={fk['L']:2d} heads={fk['heads']:.1f} ffn/d={fk['ffn_ratio']:.1f} "
          f"de={fk['d_embed']} div={fk['div_val']} ppl={math.exp(y[i]):6.2f}  res v3={R['v3'][i]:+.3f} NSC={R['NSC'][i]:+.3f}")
json.dump(dict(keys=keys, resid_v3=R['v3'].tolist(), resid_nsc=R['NSC'].tolist()),
          open(Path(__file__).with_name('failures.json'), 'w'), indent=1)

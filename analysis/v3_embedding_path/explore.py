"""V1 exploration (development split only): what explains PPL beyond model size?

Fits log(PPL) on log(size) and correlates the residual with architectural features that NSC ignores
or treats additively. Reads development labels only; the final split is not touched.
Run from project root: .venv/Scripts/python.exe analysis/v3_embedding_path/explore.py
"""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
from nsc import layers, original_nsc, resource_estimate  # noqa: E402

D = ROOT / 'nsc-starter' / 'data'
configs = json.loads((D / 'architectures.json').read_text())
res = json.loads((D / 'resources.json').read_text())
keys = json.loads((D / 'splits.json').read_text())['development']
lab = json.loads((D / 'development_labels.json').read_text())


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


rows = []
for k in keys:
    c = configs[k]
    d, heads, inner = layers(c)
    L = len(inner)
    proj = resource_estimate(c)['projection_parameters']
    rows.append(dict(
        key=k, d=d, L=L, de=c['d_embed'], div=c['div_val'],
        logppl=math.log(lab[k]['valid_ppl']), ppl=lab[k]['valid_ppl'],
        logproj=math.log(proj), lognsc=math.log(original_nsc(c)),
        logtotal=math.log(res[k]['total']), logemb=math.log(res[k]['AdaEmb']),
        attn_frac=res[k]['Attn'] / res[k]['nonembedding'],
        ffn_mean=np.mean(inner) / d, ffn_cv=np.std(inner) / np.mean(inner),
        ffn_trend=spearman(np.arange(L), inner) if L > 2 else 0.0,
        heads_mean=np.mean(heads), head_dim_mean=np.mean([d / h for h in heads]),
        aspect=math.log(d / L), de_over_d=math.log(c['d_embed'] / d)))

def col(n): return np.array([r[n] for r in rows], float)
y = col('logppl')

def fit(X, y):
    X1 = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    return beta, r, 1 - r.var() / y.var()

print('R^2 of log PPL explained by:')
for name, X in [('log proj params', [col('logproj')]),
                ('log NSC', [col('lognsc')]),
                ('log total params', [col('logtotal')]),
                ('log proj + log emb', [col('logproj'), col('logemb')]),
                ('log proj + d_embed/d', [col('logproj'), col('de_over_d')]),
                ('log proj + log d + log L', [col('logproj'), np.log(col('d')), np.log(col('L'))])]:
    b, r, r2 = fit(X, y)
    print(f'  {name:28s} R2={r2:.4f}  resid sd={r.std():.4f}  coefs={np.round(b[1:], 4)}')

_, resid, _ = fit([col('logproj')], y)
print('\nSpearman(feature, residual of log PPL ~ log proj)  [negative = feature helps]')
for f in ['d', 'L', 'aspect', 'de', 'de_over_d', 'div', 'logemb', 'attn_frac', 'ffn_mean', 'ffn_cv',
          'ffn_trend', 'heads_mean', 'head_dim_mean']:
    print(f'  {f:14s} {spearman(col(f), resid):+.3f}')

print('\nWithin-width: residual vs features (n per width)')
for w in (64, 128, 256, 512):
    idx = col('d') == w
    _, rw, r2w = fit([col('logproj')[idx]], y[idx])
    out = ' '.join(f"{f}={spearman(col(f)[idx], rw):+.2f}" for f in ['L', 'de', 'div', 'logemb', 'ffn_cv', 'heads_mean', 'attn_frac'])
    print(f'  d={w:3d} n={idx.sum():2d} R2(log proj)={r2w:.3f}  {out}')

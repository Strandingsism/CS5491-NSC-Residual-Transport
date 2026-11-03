"""V1 candidate: NSC plus spectral capacity of the (tied, adaptive) embedding matrices.

Development split only. No fitted constant: the embedding/softmax matrices are scored with the same
psi_MP(m, n, 0.02) convention as the layer matrices, counting each unique (tied) matrix once.
Shape model (validated: AdaEmb = sum of these matrices + 192*d_model for every configuration):
  cluster i covers tokens [start_i, end_i) of the cutoffs truncated at n_token, width d_i =
  d_embed / div_val^i (all d_embed when div_val=1); projection d_i x d_model unless d_i == d_model
  (div_val=1 uses a single table). tie_weight: output softmax reuses the tables; tie_projs[0]=False:
  the head cluster's output projection is a separate matrix.
Run from project root: .venv/Scripts/python.exe analysis/v1_explore/embedding_capacity.py
"""
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
from evaluate import BUDGETS, K_VALUES, canonical, correlation, pair_summary, ranks, select_trials  # noqa: E402
from nsc import original_nsc, psi_mp, resource_estimate  # noqa: E402

D = ROOT / 'nsc-starter' / 'data'


def embedding_matrices(c):
    """(rows, cols) of every unique embedding-side matrix."""
    n, d, de, div = c['n_token'], c['d_model'], c['d_embed'], c['div_val']
    ends = sorted({min(x, n) for x in c['cutoffs']})
    if div == 1:
        mats = [(n, de)] + ([(de, d)] if de != d else [])
        if not c['tie_projs'][0] and de != d:
            mats.append((de, d))
        return mats
    mats = []
    for i, (a, b) in enumerate(zip([0] + ends[:-1], ends)):
        di = de // div ** i
        mats += [(b - a, di), (di, d)]
        if i == 0 and not c['tie_projs'][0]:
            mats.append((di, d))
    return mats


def embedding_capacity(c):
    return math.fsum(psi_mp(m, k) for m, k in embedding_matrices(c))


def nsc_with_embedding(c):
    return original_nsc(c) + embedding_capacity(c)


def main():
    configs = json.loads((D / 'architectures.json').read_text())
    res = json.loads((D / 'resources.json').read_text())
    keys = json.loads((D / 'splits.json').read_text())['development']
    lab = json.loads((D / 'development_labels.json').read_text())
    ppl = [lab[k]['valid_ppl'] for k in keys]
    q = [-p for p in ppl]
    total = [res[k]['total'] for k in keys]
    methods = {
        'nsc': original_nsc,
        'nsc+emb (V1)': nsc_with_embedding,
        'emb only': embedding_capacity,
        'proj params': lambda c: resource_estimate(c)['projection_parameters'],
        'proj+emb params': lambda c: resource_estimate(c)['projection_parameters'] + res_emb(c),
    }
    global res_emb
    res_emb = lambda c: sum(m * k for m, k in embedding_matrices(c))
    scores = {n: [canonical(f(configs[k])) for k in keys] for n, f in methods.items()}
    scores['total params'] = total
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
    k0 = keys[0]
    print('example', configs[k0]['d_model'], configs[k0]['d_embed'], configs[k0]['div_val'],
          'NSC=%.1f emb=%.1f' % (original_nsc(configs[k0]), embedding_capacity(configs[k0])))
    print(f"{'score':16s} {'rho':>6s} {'tau':>6s} {'ctl':>6s}  within-width rho d64/128/256/512")
    for n, s in scores.items():
        ww = []
        for w in (64, 128, 256, 512):
            idx = [i for i, k in enumerate(keys) if configs[k]['d_model'] == w]
            ww.append(correlation(ranks([s[i] for i in idx]), ranks([q[i] for i in idx])))
        print(f"{n:16s} {correlation(ranks(s), ranks(q)):6.3f} {pair_summary(s, q, pairs)['tau_b']:6.3f} "
              f"{pair_summary(s, q, matched)['tau_b']:6.3f}  " + ' '.join(f'{v:.3f}' for v in ww))
    print('\nmean regret by cap/k')
    for n, s in scores.items():
        out = []
        for b in BUDGETS:
            pool = [i for i, t in enumerate(total) if t <= b]
            for k in K_VALUES:
                out.append(select_trials(pool, s, k, ppl, 202627 + b + k)['mean_regret'])
        print(f"{n:16s} " + ' '.join(f'{v:6.2f}' for v in out))


if __name__ == '__main__':
    main()


def embedding_tables(c):
    """(tokens, width) of each adaptive-embedding table (shared with the tied output softmax)."""
    n, de, div = c['n_token'], c['d_embed'], c['div_val']
    if div == 1:
        return [(n, de)]
    ends = sorted({min(x, n) for x in c['cutoffs']})
    return [(b - a, de // div ** i) for i, (a, b) in enumerate(zip([0] + ends[:-1], ends))]


def bottleneck_embedding_capacity(c):
    """Each table contributes at most min(width, d_model) channels: its path into the residual
    stream passes through a width x d_model projection (rank bottleneck)."""
    return math.fsum(psi_mp(v, min(w, c['d_model'])) for v, w in embedding_tables(c))


def nsc_bottleneck(c):
    return original_nsc(c) + bottleneck_embedding_capacity(c)


def path_embedding_capacity(c, sigma=0.02):
    """Capacity of each table's path E P into the residual stream under Post-LN scale invariance.

    E (v x w) has N(0, sigma^2) entries; the projection P (w x d) is taken variance-preserving,
    N(0, 1/w), because LN removes its overall scale. With v >> w (w/v <= 0.05 in the panel),
    E^T E ~= sigma^2 v I_w, so the nonzero spectrum of (E P)^T (E P) is that of a w x d matrix with
    entries N(0, sigma^2 v / w): psi_MP(w, d, sigma * sqrt(v / w)). A div_val=1 table with
    d_embed == d_model has no projection and is scored directly as psi_MP(v, d, sigma).
    """
    d = c['d_model']
    total = []
    for v, w in embedding_tables(c):
        if c['div_val'] == 1 and w == d:
            total.append(psi_mp(v, d, sigma))
        else:
            total.append(psi_mp(w, d, sigma * math.sqrt(v / w)))
    return math.fsum(total)


def nsc_path(c):
    return original_nsc(c) + path_embedding_capacity(c)


def series_score(c, passes=2):
    """Series composition of the embedding stage(s) with the residual stack.

    Residual branches are parallel paths (y = x + sum_s F_s), so layer capacities add (NSC). Every
    path passes through the embedding table on the way in and, with tied weights, again at the
    output softmax: stages in series combine like resistances, 1/S = 1/NSC + passes/E.
    passes=2 counts the tied table per invocation (primary); passes=1 counts it once (sensitivity).
    """
    n, e = original_nsc(c), path_embedding_capacity(c)
    return 1.0 / (1.0 / n + passes / e)

"""Single development evaluation of the path-capacity V1 score (decided before this run).

Compares NSC (ablation), the earlier cap version, and the path version on ranking, size-controlled
pairs, within-width rho, selection regret, and a bootstrap over architectures. Development only.
Run from project root: .venv/Scripts/python.exe analysis/v3_embedding_path/dev_eval_path.py
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import BUDGETS, K_VALUES, canonical, correlation, pair_summary, ranks, select_trials  # noqa: E402
from nsc import original_nsc, position_example, resource_estimate  # noqa: E402
from embedding_capacity import nsc_bottleneck, nsc_path  # noqa: E402

D = ROOT / 'nsc-starter' / 'data'
configs = json.loads((D / 'architectures.json').read_text())
res = json.loads((D / 'resources.json').read_text())
keys = json.loads((D / 'splits.json').read_text())['development']
lab = json.loads((D / 'development_labels.json').read_text())
ppl = [lab[k]['valid_ppl'] for k in keys]
q = [-p for p in ppl]
total = [res[k]['total'] for k in keys]
W = [configs[k]['d_model'] for k in keys]
METHODS = {'NSC (ablation)': original_nsc, 'V1 cap (earlier)': nsc_bottleneck, 'V1 path (chosen)': nsc_path,
           'l/L example': position_example,
           'proj params': lambda c: resource_estimate(c)['projection_parameters']}
scores = {n: [canonical(f(configs[k])) for k in keys] for n, f in METHODS.items()}
scores['total params'] = total
pairs = [(i, j) for i in range(100) for j in range(i)]
matched = [(i, j) for i, j in pairs if W[i] == W[j] and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
out = {}
print(f"{'score':18s} {'rho':>7s} {'tau_b':>7s} {'ctl':>6s}  within-width d64/128/256/512   regret 10/20/30/50M x k1,k3")
for n, s in scores.items():
    ww = [correlation(ranks([s[i] for i in range(100) if W[i] == w]), ranks([q[i] for i in range(100) if W[i] == w]))
          for w in (64, 128, 256, 512)]
    reg = []
    for b in BUDGETS:
        pool = [i for i, t in enumerate(total) if t <= b]
        for k in K_VALUES:
            reg.append(select_trials(pool, s, k, ppl, 202627 + b + k)['mean_regret'])
    rho, tau, ctl = correlation(ranks(s), ranks(q)), pair_summary(s, q, pairs)['tau_b'], pair_summary(s, q, matched)['tau_b']
    out[n] = dict(rho=rho, tau_b=tau, controlled_tau_b=ctl, within_width=ww, regret=reg)
    print(f'{n:18s} {rho:7.4f} {tau:7.4f} {ctl:6.3f}  ' + ' '.join(f'{v:.3f}' for v in ww) + '   ' + ' '.join(f'{v:.2f}' for v in reg))

rng = np.random.default_rng(5491)
qa = np.array(q)
boot = {}
for a, b in [('V1 path (chosen)', 'NSC (ablation)'), ('V1 path (chosen)', 'V1 cap (earlier)')]:
    sa, sb, d = np.array(scores[a]), np.array(scores[b]), []
    for _ in range(5000):
        i = rng.integers(0, 100, 100)
        d.append(correlation(ranks(list(sa[i])), ranks(list(qa[i]))) - correlation(ranks(list(sb[i])), ranks(list(qa[i]))))
    d = np.array(d)
    boot[f'{a} - {b}'] = dict(mean=float(d.mean()), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], p_le_0=float((d <= 0).mean()))
    print(f'bootstrap rho[{a}] - rho[{b}]: {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  P(<=0)={(d <= 0).mean():.4f}')
(Path(__file__).with_name('dev_eval_path.json')).write_text(json.dumps(dict(methods=out, bootstrap=boot, matched_pairs=len(matched)), indent=1) + '\n', encoding='utf-8')

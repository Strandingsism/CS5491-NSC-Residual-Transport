"""Pre-registered evaluation of frozen v3 (tag v3-frozen) on one split.

Primary: Spearman rho of v3 versus original NSC (its ablation) on negative PPL, with a bootstrap over
architectures (5000 resamples, seed 5491) for rho(v3) - rho(NSC). Secondary: Kendall tau-b,
size-controlled tau-b, within-width rho, and the starter's selection regret. Written and committed
before the final split is evaluated for v3; run on --split final exactly once.

Run from project root:
  .venv/Scripts/python.exe analysis/v3_embedding_path/v3_final_eval.py --split development
  .venv/Scripts/python.exe analysis/v3_embedding_path/v3_final_eval.py --split final
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
from evaluate import BUDGETS, K_VALUES, canonical, correlation, pair_summary, ranks, select_trials  # noqa: E402
from nsc import original_nsc, position_example, resource_estimate  # noqa: E402
import v3_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'final'), required=True)
    split = parser.parse_args().split
    label = 'valid_ppl' if split == 'development' else 'test_ppl'
    D = ROOT / 'nsc-starter' / 'data'
    configs = json.loads((D / 'architectures.json').read_text())
    res = json.loads((D / 'resources.json').read_text())
    keys = json.loads((D / 'splits.json').read_text())[split]
    lab = json.loads((D / f'{split}_labels.json').read_text())
    ppl = [lab[k][label] for k in keys]
    q = [-p for p in ppl]
    total = [res[k]['total'] for k in keys]
    W = [configs[k]['d_model'] for k in keys]
    n = len(keys)
    methods = {'v3': v3_score.score, 'NSC (ablation)': v3_score.ablation, 'l/L example': position_example,
               'projection params': lambda c: resource_estimate(c)['projection_parameters'],
               'reference FLOPs': lambda c: resource_estimate(c)['reference_flops_per_token']}
    scores = {m: [canonical(f(configs[k])) for k in keys] for m, f in methods.items()}
    scores['total params'] = total
    pairs = [(i, j) for i in range(n) for j in range(i)]
    matched = [(i, j) for i, j in pairs if W[i] == W[j] and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
    out = dict(split=split, label=label, matched_pairs=len(matched), methods={})
    print(f"{split}: {n} architectures, {len(matched)} size-controlled pairs")
    print(f"{'score':18s} {'rho':>7s} {'tau_b':>7s} {'ctl':>6s}  within-width d64/128/256/512   regret 10/20/30/50M x k1,k3")
    for m, s in scores.items():
        ww = [correlation(ranks([s[i] for i in range(n) if W[i] == w]), ranks([q[i] for i in range(n) if W[i] == w]))
              for w in (64, 128, 256, 512)]
        reg = []
        for b in BUDGETS:
            pool = [i for i, t in enumerate(total) if t <= b]
            for k in K_VALUES:
                reg.append(select_trials(pool, s, k, ppl, 202627 + b + k)['mean_regret'])
        r = dict(rho=correlation(ranks(s), ranks(q)), tau_b=pair_summary(s, q, pairs)['tau_b'],
                 controlled_tau_b=pair_summary(s, q, matched)['tau_b'], within_width=ww, regret=reg)
        out['methods'][m] = r
        print(f"{m:18s} {r['rho']:7.4f} {r['tau_b']:7.4f} {r['controlled_tau_b']:6.3f}  "
              + ' '.join(f'{v:.3f}' for v in ww) + '   ' + ' '.join(f'{v:.2f}' for v in reg))
    rng = np.random.default_rng(5491)
    a, b, qa, d = np.array(scores['v3']), np.array(scores['NSC (ablation)']), np.array(q), []
    for _ in range(5000):
        i = rng.integers(0, n, n)
        d.append(correlation(ranks(list(a[i])), ranks(list(qa[i]))) - correlation(ranks(list(b[i])), ranks(list(qa[i]))))
    d = np.array(d)
    out['bootstrap_rho_v3_minus_nsc'] = dict(mean=float(d.mean()), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                                             p_le_0=float((d <= 0).mean()))
    print(f"bootstrap rho(v3) - rho(NSC): {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, "
          f"{np.percentile(d, 97.5):+.4f}]  P(<=0)={(d <= 0).mean():.4f}")
    (Path(__file__).with_name(f'v3_{split}_eval.json')).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

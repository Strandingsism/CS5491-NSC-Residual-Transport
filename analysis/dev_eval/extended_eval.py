"""Evaluation with the proposal's additional baselines and controls, for one split.

Reads ONLY the chosen split's labels (as evaluate.py does). Ranking/selection statistics
reuse the starter's evaluate.py functions. Also times specification-only scoring of the
panel, cold (empty MP/transport caches) and warm.

Run from project root:
  .venv/Scripts/python.exe analysis/dev_eval/extended_eval.py                 # development
  .venv/Scripts/python.exe analysis/dev_eval/extended_eval.py --split final   # after freezing
"""
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
STARTER = ROOT / 'nsc-starter'
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(ROOT / 'analysis' / 'postln_transport'))
from evaluate import BUDGETS, K_VALUES, canonical, correlation, pair_summary, ranks, select_trials  # noqa: E402
from nsc import layers, original_nsc, position_example, psi_mp, resource_estimate  # noqa: E402
from residual_transport import causal_coefficients, transport_factors  # noqa: E402
import student_score  # noqa: E402
import transport as v1  # noqa: E402


def shape_only(config):
    """Post-LN weights rescaled to mean 1 within the configuration: isolates the position/
    allocation profile from the configuration-level scale (diagnostic, not a frozen method)."""
    d, _, inner = layers(config)
    T = transport_factors(d, inner, 'post', config['tgt_len'], config['mem_len'])
    mean = sum(T) / len(T)
    return math.fsum(t / mean * c for t, c in zip(T, student_score.sublayer_capacities(config)))


METHODS = {
    'student_post_v2': student_score.score,
    'ablation_original_nsc': student_score.ablation,
    'position_example_l_over_L': position_example,
    'post_v1_iid_tokens': lambda c: v1.transport_score(c, 'post'),
    'post_v2_shape_only': shape_only,
    'pre_v2_mismatched_norm': lambda c: student_score.transport_weighted_nsc(c, 'pre'),
    'none_v2_no_norm': lambda c: student_score.transport_weighted_nsc(c, 'none'),
    'depth': lambda c: c['n_layer'],
    'projection_parameters': lambda c: resource_estimate(c)['projection_parameters'],
    'reference_flops': lambda c: resource_estimate(c)['reference_flops_per_token'],
}


def scoring_seconds(configs, keys, repeats=5):
    """Wall seconds to score the split: cold (caches cleared) and warm, median of repeats."""
    out = {}
    for name in ('student_post_v2', 'ablation_original_nsc'):
        cold, warm = [], []
        for _ in range(repeats):
            psi_mp.cache_clear()
            causal_coefficients.cache_clear()
            t = time.perf_counter()
            for k in keys:
                METHODS[name](configs[k])
            cold.append(time.perf_counter() - t)
            t = time.perf_counter()
            for k in keys:
                METHODS[name](configs[k])
            warm.append(time.perf_counter() - t)
        out[name] = dict(cold_median=sorted(cold)[repeats // 2], warm_median=sorted(warm)[repeats // 2])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'final'), default='development')
    split = parser.parse_args().split
    label_name = 'valid_ppl' if split == 'development' else 'test_ppl'
    data = STARTER / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    resources = json.loads((data / 'resources.json').read_text())
    keys = json.loads((data / 'splits.json').read_text())[split]
    timing = scoring_seconds(configs, keys)
    labels = json.loads((data / f'{split}_labels.json').read_text())
    ppl = [labels[k][label_name] for k in keys]
    quality = [-v for v in ppl]
    total = [resources[k]['total'] for k in keys]
    scores = {name: [canonical(fn(configs[k])) for k in keys] for name, fn in METHODS.items()}
    scores['reported_parameters'] = total

    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
    widths = sorted({configs[k]['d_model'] for k in keys})
    nsc = scores['ablation_original_nsc']

    ranking = {}
    for name, s in scores.items():
        within = {}
        for w in widths:
            idx = [i for i, k in enumerate(keys) if configs[k]['d_model'] == w]
            within[str(w)] = dict(n=len(idx), spearman_rho=correlation(ranks([s[i] for i in idx]),
                                                                        ranks([quality[i] for i in idx])))
        vs = pair_summary(s, nsc, pairs)
        ranking[name] = dict(spearman_rho=correlation(ranks(s), ranks(quality)),
                             all_pairs=pair_summary(s, quality, pairs),
                             size_controlled_pairs=pair_summary(s, quality, matched),
                             within_width=within,
                             vs_original_nsc=dict(tau_b=vs['tau_b'], discordant_pairs=vs['discordant'],
                                                  matched_discordant=pair_summary(s, nsc, matched)['discordant']))
    selection = []
    for budget in BUDGETS:
        pool = [i for i, t in enumerate(total) if t <= budget]
        for k in K_VALUES:
            if len(pool) < k:
                continue
            seed = 202627 + budget + k
            res = {name: select_trials(pool, s, k, ppl, seed) for name, s in scores.items()}
            res['random'] = select_trials(pool, None, k, ppl, seed)
            selection.append(dict(parameter_cap=budget, k=k, eligible=len(pool),
                                  oracle_ppl=min(ppl[i] for i in pool), methods=res))
    result = dict(split=split, label=label_name, architectures=len(keys),
                  matched_pairs=len(matched), ranking=ranking, selection=selection,
                  scoring_seconds=timing,
                  hardware=dict(processor=platform.processor(), python=platform.python_version()))
    (OUT / f'{split}_extended.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    f = lambda v: '   n/a' if v is None else f'{v:+.4f}'
    print(f"{'method':28s} {'rho':>7s} {'tau_b':>7s} {'ctl_tau':>7s} {'disc_vs_NSC':>11s} {'ties':>7s} "
          + ' '.join(f'rho_d{w}' for w in widths))
    for name, r in ranking.items():
        a, m = r['all_pairs'], r['size_controlled_pairs']
        ties = f"{a['score_only_ties'] + a['both_ties']}/{m['score_only_ties'] + m['both_ties']}"
        print(f"{name:28s} {f(r['spearman_rho'])} {f(r['all_pairs']['tau_b'])} "
              f"{f(r['size_controlled_pairs']['tau_b'])} {r['vs_original_nsc']['discordant_pairs']:>5d}/"
              f"{r['vs_original_nsc']['matched_discordant']:<4d} {ties:>7s} "
              + ' '.join(f(r['within_width'][str(w)]['spearman_rho']) for w in widths))
    print(f'matched pairs: {len(matched)}')
    names = list(scores) + ['random']
    print('\nmean regret (PPL) by cap/k')
    print(f"{'method':28s} " + ' '.join(f"{s['parameter_cap'] // 10**6}M/k{s['k']}".rjust(8) for s in selection))
    for name in names:
        print(f'{name:28s} ' + ' '.join(f"{s['methods'][name]['mean_regret']:8.3f}" for s in selection))
    print('\nmean best selected PPL by cap/k')
    print(f"{'oracle (best in pool)':28s} " + ' '.join(f"{s['oracle_ppl']:8.3f}" for s in selection))
    for name in names:
        print(f'{name:28s} ' + ' '.join(f"{s['methods'][name]['mean_best_ppl']:8.3f}" for s in selection))
    print('\nscoring seconds for the split (median of 5):', json.dumps(timing))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate ranking and fixed-budget selection using archived trained outcomes.

Run from any directory: python3 evaluate.py --split development --output dev.json
Freeze your code and choices before running --split final.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import time

from nsc import original_nsc, position_example, resource_estimate

ROOT = Path(__file__).resolve().parent
BUDGETS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)
K_VALUES = (1, 3)
REPEATS = 200


def ranks(values):
    """Average one-based ranks, with exact ties after score canonicalization."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            result[k] = (i + 1 + j) / 2
        i = j
    return result


def correlation(x, y):
    """Pearson correlation; null if either vector is constant or too short."""
    if len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    xx, yy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in xx) * sum(v * v for v in yy))
    return sum(a * b for a, b in zip(xx, yy)) / denom if denom else None


def pair_summary(scores, quality, pairs):
    """Tau-b and tie counts on the stated set of pairs (no independence claim)."""
    c = d = tx = ty = both = 0
    for i, j in pairs:
        dx = (scores[i] > scores[j]) - (scores[i] < scores[j])
        dy = (quality[i] > quality[j]) - (quality[i] < quality[j])
        if dx == 0 and dy == 0:
            both += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            c += 1
        else:
            d += 1
    denom = math.sqrt((c + d + tx) * (c + d + ty))
    return {'pairs': c + d + tx + ty + both, 'concordant': c, 'discordant': d,
            'score_only_ties': tx, 'quality_only_ties': ty, 'both_ties': both,
            'tau_b': (c - d) / denom if denom else None}


def canonical(value):
    """Use 12 significant digits so numerical noise does not break score ties."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('All scores must be finite')
    return float(format(value, '.12g'))


def select_trials(pool, scores, k, ppl, seed):
    """Best revealed PPL after choosing k designs; randomize only score ties.

    scores=None gives uniform random selection without replacement.
    Variability measures random selection/tie-breaking, NOT training noise.
    """
    rng = random.Random(seed)
    best = []
    for _ in range(REPEATS):
        if scores is None:
            chosen = rng.sample(pool, k)
        else:
            order = list(pool)
            rng.shuffle(order)
            order.sort(key=lambda i: scores[i], reverse=True)
            chosen = order[:k]
        best.append(min(ppl[i] for i in chosen))
    oracle = min(ppl[i] for i in pool)
    return {'mean_best_ppl': statistics.mean(best),
            'mean_regret': statistics.mean(best) - oracle,
            'sd_best_ppl': statistics.pstdev(best), 'trials': REPEATS}


def evaluate(split, student_path):
    """Load only the chosen split's labels and return JSON-serializable results."""
    def read(name):
        return json.loads((ROOT / 'data' / name).read_text())
    configs = read('architectures.json')
    resources = read('resources.json')
    keys = read('splits.json')[split]
    label_name = 'valid_ppl' if split == 'development' else 'test_ppl'
    labels = read(split + '_labels.json')
    ppl = [labels[k][label_name] for k in keys]
    quality = [-v for v in ppl]
    spec = importlib.util.spec_from_file_location('student_submission', student_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    methods = {'original_nsc': original_nsc, 'position_example': position_example,
               'student': module.score, 'ablation': module.ablation}
    scores, timings = {}, {}
    for name, function in methods.items():
        start = time.perf_counter()
        scores[name] = [canonical(function(copy.deepcopy(configs[k]))) for k in keys]
        timings[name] = time.perf_counter() - start
    # Use the archive's total field directly: component fields overlap.
    scores['reported_parameters'] = [resources[k]['total'] for k in keys]
    scores['reference_flops'] = [resource_estimate(configs[k])['reference_flops_per_token'] for k in keys]
    total = scores['reported_parameters']
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
    ranking = {}
    for name, values in scores.items():
        ranking[name] = {'spearman_rho': correlation(ranks(values), ranks(quality)),
                         'all_pairs': pair_summary(values, quality, pairs),
                         'size_controlled_pairs': pair_summary(values, quality, matched)}
    selection = []
    for budget in BUDGETS:
        pool = [i for i, size in enumerate(total) if size <= budget]
        for k in K_VALUES:
            if len(pool) < k:
                continue
            seed = 202627 + budget + k
            results = {name: select_trials(pool, values, k, ppl, seed)
                       for name, values in scores.items()}
            results['random'] = select_trials(pool, None, k, ppl, seed)
            selection.append({'parameter_cap': budget, 'k': k, 'eligible': len(pool),
                              'oracle_ppl': min(ppl[i] for i in pool), 'methods': results})
    return {'split': split, 'label': label_name, 'architectures': len(keys),
            'score_precision': '12 significant digits',
            'matched_rule': 'same d_model; min(total)/max(total)>=0.90',
            'timing_note': 'Wall seconds for one panel; shared MP cache and method order affect timings. Benchmark cold/warm scoring separately for runtime claims.',
            'scoring_seconds': timings, 'ranking': ranking, 'selection': selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'final'), default='development')
    parser.add_argument('--student', type=Path, default=ROOT / 'student_score.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.split, args.student.resolve())
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload)
    for name, stats in result['ranking'].items():
        fmt = lambda value: 'undefined' if value is None else f'{value:.6f}'
        print(f"{name:20s} rho={fmt(stats['spearman_rho']):>10s} "
              f"tau={fmt(stats['all_pairs']['tau_b']):>10s} "
              f"controlled_tau={fmt(stats['size_controlled_pairs']['tau_b']):>10s}")
    print(f"{result['architectures']} {args.split} architectures; "
          f"{len(result['selection'])} budget/k combinations")
    if args.output:
        print(f'Full results: {args.output}')


if __name__ == '__main__':
    main()

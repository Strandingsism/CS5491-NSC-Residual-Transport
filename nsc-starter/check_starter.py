#!/usr/bin/env python3
"""Independent numerical/invariant checks; does not read final outcome labels."""
import copy
import hashlib
import json
import math
from pathlib import Path

from evaluate import pair_summary, ranks, correlation, select_trials
from nsc import original_nsc, position_example, psi_mp, resource_estimate

ROOT = Path(__file__).resolve().parent


def main():
    # MP at zero power, transpose invariance, small-signal first moment,
    # and integration convergence. Bounds from log(1+x)<=x and Jensen.
    for m, n in [(1, 1), (64, 64), (128, 32), (512, 1601)]:
        p = psi_mp(m, n)
        assert psi_mp(m, n, 0) == 0
        assert p == psi_mp(n, m)
        assert math.isclose(p, psi_mp(m, n, steps=1024), rel_tol=1e-10)
        assert 0 < p <= min(m, n) * math.log1p(max(m, n) * .02**2)
        assert math.isclose(psi_mp(m, n, 1e-6), m * n * 1e-12, rel_tol=1e-7)
    # Analytic scalar identity at the MP square case: derive the derivative
    # of the log integral or use the closed-form Shannon transform.
    a = 64 * .02**2
    root = math.sqrt(1 + 4 * a)
    exact_square = 64 * (2 * math.log((1 + root) / 2) + 2 / (1 + root) - 1)
    assert math.isclose(psi_mp(64, 64), exact_square, rel_tol=1e-11)
    cfg = {'d_model': 128, 'n_layer': 2, 'n_head': [4, 4], 'd_inner': [256, 512]}
    reverse = copy.deepcopy(cfg)
    reverse['d_inner'].reverse()
    assert math.isclose(original_nsc(cfg), original_nsc(reverse), rel_tol=1e-14)
    assert position_example(cfg) != position_example(reverse)
    assert resource_estimate(cfg) == resource_estimate(reverse)
    assert resource_estimate(cfg)['projection_parameters'] == 327680
    assert resource_estimate(cfg)['reference_macs_per_token'] == 524288
    assert ranks([3, 1, 1]) == [3.0, 1.5, 1.5]
    assert correlation(ranks([1, 2, 3]), ranks([3, 2, 1])) == -1
    pairs = [(1, 0), (2, 0), (2, 1)]
    assert pair_summary([1, 2, 3], [1, 2, 3], pairs)['tau_b'] == 1
    assert pair_summary([1, 1, 1], [1, 2, 3], pairs)['tau_b'] is None
    assert math.isclose(pair_summary([1, 1, 2], [1, 2, 3], pairs)['tau_b'], 2 / math.sqrt(6))
    assert select_trials([0, 1, 2], [3, 2, 1], 1, [10, 20, 30], 1)['mean_best_ppl'] == 10
    assert select_trials([0, 1, 2], None, 3, [10, 20, 30], 1)['mean_regret'] == 0
    data = ROOT / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    splits = json.loads((data / 'splits.json').read_text())
    assert len(splits['development']) == len(splits['final']) == 100
    assert not set(splits['development']) & set(splits['final'])
    assert set(splits['development']) | set(splits['final']) == set(configs)
    for config in configs.values():
        assert math.isfinite(original_nsc(config))
        assert config['weight_init_std'] == .02
    for key, expected in [('config_0_j0', 589.8681648717517),
                          ('config_0_j1', 12950.414017875013)]:
        assert math.isclose(original_nsc(configs[key]), expected, rel_tol=1e-8)
    # File integrity, without parsing or reporting final labels.
    provenance = json.loads((data / 'provenance.json').read_text())
    for filename, expected in provenance['hashes'].items():
        assert hashlib.sha256((data / filename).read_bytes()).hexdigest() == expected
    print('PASS: MP identities/convergence, reorder controls, resource arithmetic,')
    print('rank ties, selection, 200 configurations, split separation and data hashes.')
    print(f"Toy original NSC: {original_nsc(cfg):.9f} (both orders)")
    print(f"Toy position example: {position_example(cfg):.9f} vs {position_example(reverse):.9f}")


if __name__ == '__main__':
    main()

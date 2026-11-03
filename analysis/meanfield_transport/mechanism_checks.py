"""Label-free mechanism checks for the transport-weighted score.

1. Toy reorder pair (d=128, H=4, FFN [256,512] vs [512,256]) under post/pre/none.
2. Weight profiles on the development panel (archive settings, n=mem=192) and how far each
   weighting moves the development ranking away from original NSC (no PPL used).
3. Stronger-signal regime: sigma raised so branch gains are O(0.1-1); MC vs prediction.
   This tests the recursion where its weights are far from uniform; it is not the archive.

Run from project root: .venv/Scripts/python.exe analysis/meanfield_transport/mechanism_checks.py
"""
from pathlib import Path
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
sys.path.insert(0, str(OUT))
from evaluate import pair_summary  # noqa: E402
from nsc import layers, original_nsc, position_example  # noqa: E402
from residual_transport import transport_factors  # noqa: E402
import student_score  # noqa: E402
import meanfield as M  # noqa: E402


def toy():
    a = {'d_model': 128, 'n_layer': 2, 'n_head': [4, 4], 'd_inner': [256, 512], 'tgt_len': 192, 'mem_len': 192}
    b = dict(a, d_inner=[512, 256])
    return {norm: dict(ffn_256_512=student_score.transport_weighted_nsc(a, norm),
                       ffn_512_256=student_score.transport_weighted_nsc(b, norm))
            for norm in ('post', 'pre', 'none')} | {
        'original_nsc': [original_nsc(a), original_nsc(b)], 'position_example': [position_example(a), position_example(b)]}


def panel(configs, keys, totals):
    out = {}
    nsc = [original_nsc(configs[k]) for k in keys]
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(totals[i], totals[j]) / max(totals[i], totals[j]) >= 0.90]
    for norm in ('post', 'pre', 'none'):
        spread, trend, lo, hi = [], [], 1e9, 0
        for k in keys:
            d, _, inner = layers(configs[k])
            T = np.array(transport_factors(d, inner, norm, 192, 192))
            spread.append(T.max() / T.min())
            lo, hi = min(lo, T.min()), max(hi, T.max())
            if len(T) > 2:
                trend.append(np.corrcoef(np.argsort(np.argsort(T)), np.arange(len(T)))[0, 1])
        s = [student_score.transport_weighted_nsc(configs[k], norm) for k in keys]
        topk = {}
        for cap in (10_000_000, 20_000_000, 30_000_000, 50_000_000):
            pool = [i for i, t in enumerate(totals) if t <= cap]
            for kk in (1, 3):
                a = set(sorted(pool, key=lambda i: -nsc[i])[:kk])
                b = set(sorted(pool, key=lambda i: -s[i])[:kk])
                topk[f'{cap // 10**6}M_k{kk}'] = len(a ^ b) // 2
        out[norm] = dict(T_min=float(lo), T_max=float(hi),
                         within_max_over_min_median=float(np.median(spread)),
                         within_max_over_min_max=float(max(spread)),
                         position_rank_trend_mean=float(np.mean(trend)),
                         discordant_vs_nsc=pair_summary(s, nsc, pairs)['discordant'],
                         matched_discordant_vs_nsc=pair_summary(s, nsc, matched)['discordant'],
                         topk_changed_members=topk)
    return out


def strong_regime():
    rows = []
    for sigma, d, f, L in [(0.04, 128, 512, 8), (0.05, 128, 512, 6), (0.03, 256, 1024, 8)]:
        for norm in ('post', 'pre', 'none'):
            M.SIGMA = sigma  # make_sublayers default reads module SIGMA at call time below
            rng_T = []
            for seed in range(4):
                rng = np.random.default_rng(seed)
                subs = M.make_sublayers(d, [4] * L, [f] * L, rng, sigma)
                x0 = rng.normal(size=(64, d))
                rec = []
                M.run(x0, subs, norm, None, record=rec)
                T = []
                for s in range(2 * L):
                    acc = 0.0
                    for _ in range(2):
                        v = rng.normal(size=(64, d))
                        diff = M.run(rec[s], subs, norm, None, s, 1e-4 * v) - M.run(rec[s], subs, norm, None, s, -1e-4 * v)
                        acc += (diff * diff).sum() / 4e-8 / (64 * d) / 2
                    T.append(acc)
                rng_T.append(T)
            meas = np.mean(rng_T, 0)
            pred = np.array(M.predict_meanfield(d, [f] * L, norm, 64, 0, sigma))
            rows.append(dict(sigma=sigma, d=d, f=f, L=L, norm=norm,
                             measured=meas.round(4).tolist(), predicted=pred.round(4).tolist(),
                             rel_error_rms=float(np.sqrt(np.mean((pred / meas - 1) ** 2))),
                             measured_max_over_min=float(meas.max() / meas.min()),
                             predicted_max_over_min=float(pred.max() / pred.min())))
    M.SIGMA = 0.02
    return rows


def main():
    data = ROOT / 'nsc-starter' / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    keys = json.loads((data / 'splits.json').read_text())['development']
    res = json.loads((data / 'resources.json').read_text())
    totals = [res[k]['total'] for k in keys]
    result = dict(toy_reorder=toy(), development_panel=panel(configs, keys, totals), strong_regime=strong_regime())
    (OUT / 'mechanism_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'strong_regime'}, indent=1))
    for r in result['strong_regime']:
        print(r['sigma'], r['d'], r['f'], r['L'], r['norm'], f"rms err {r['rel_error_rms']:.3f}",
              f"spread meas {r['measured_max_over_min']:.2f} pred {r['predicted_max_over_min']:.2f}")


if __name__ == '__main__':
    main()

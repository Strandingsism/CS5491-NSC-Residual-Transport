"""V1 mechanism checks (label-free): is the bottleneck cap the capacity of the embedding path?

1. Path capacity. A token's embedding reaches the residual stream as e^T P with table E (v x w,
   entries N(0, s^2)) and projection P (w x d). Post-LN removes the overall scale of P (v2 result),
   so we take a variance-preserving P~ with entries N(0, 1/w) and measure
       psi_path = E sum_i ln(1 + s_i(E P~)^2)
   by Monte Carlo: (E P~)^T (E P~) = P~^T (E^T E) P~ with E^T E ~ s^2 Wishart_w(v, I), so no v x w
   matrix is formed. Compared with the cap psi_MP(v, min(w, d)) for every table shape in the panel.
   The literal-scale product (P entries N(0, s^2)) is reported too: it is negligible, which is why
   the score must rely on scale invariance rather than on raw product spectra.
2. Saturation pattern per width and d_embed from the capacities alone (no labels).
3. Controlled pair: identical layers, different d_embed.

Run from project root: .venv/Scripts/python.exe analysis/v3_embedding_path/mechanism_check.py
"""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsc import original_nsc, psi_mp, resource_estimate  # noqa: E402
from embedding_capacity import bottleneck_embedding_capacity, embedding_capacity, embedding_tables  # noqa: E402

SIGMA = 0.02
SEEDS = 6


def path_capacity(v, w, d, rng, proj_var=None):
    """MC mean of sum ln(1 + s^2) for singular values s of E P; proj_var None = variance preserving."""
    var = 1.0 / w if proj_var is None else proj_var
    out = []
    for _ in range(SEEDS):
        g = rng.standard_normal((v, w)) if v * w <= 4_000_000 else None
        if g is not None:
            gram = SIGMA ** 2 * (g.T @ g)
        else:  # Wishart via Bartlett-free route: accumulate in row blocks
            gram = np.zeros((w, w))
            for start in range(0, v, 20000):
                blk = rng.standard_normal((min(20000, v - start), w))
                gram += blk.T @ blk
            gram *= SIGMA ** 2
        p = rng.standard_normal((w, d)) * math.sqrt(var)
        ev = np.linalg.eigvalsh(p.T @ gram @ p)
        out.append(float(np.sum(np.log1p(np.clip(ev, 0, None)))))
    return float(np.mean(out)), float(np.std(out, ddof=1) / math.sqrt(SEEDS))


def main():
    configs = json.loads((ROOT / 'nsc-starter' / 'data' / 'architectures.json').read_text())
    rng = np.random.default_rng(5491)
    shapes = sorted({(v, w, c['d_model']) for c in configs.values() for v, w in embedding_tables(c)})
    print(f'1. path capacity vs cap, {len(shapes)} distinct (tokens, width, d_model) shapes in the panel')
    print(f"{'v':>6s} {'w':>4s} {'d':>4s} {'cap psi(v,min)':>14s} {'MC path':>10s} {'s.e.':>6s} {'rel err':>8s} {'literal':>8s}")
    rows = []
    for v, w, d in shapes:
        cap = psi_mp(v, min(w, d))
        mc, se = path_capacity(v, w, d, rng)
        lit, _ = path_capacity(v, w, d, rng, proj_var=SIGMA ** 2)
        rows.append(dict(v=v, w=w, d=d, cap=cap, mc=mc, se=se, literal=lit))
        print(f'{v:6d} {w:4d} {d:4d} {cap:14.2f} {mc:10.2f} {se:6.2f} {mc / cap - 1:+8.3f} {lit:8.3f}')
    rel = np.array([r['mc'] / r['cap'] - 1 for r in rows])
    print(f'   rel err: mean {rel.mean():+.3f}, max |.| {np.abs(rel).max():.3f}; '
          f'w>=d: {np.abs(rel[[r["w"] >= r["d"] for r in rows]]).max():.3f}, '
          f'w<d: {np.abs(rel[[r["w"] < r["d"] for r in rows]]).max():.3f}')

    # per-config: total MC path capacity vs bottleneck_embedding_capacity (table lookup from shapes)
    lut = {(r['v'], r['w'], r['d']): r for r in rows}
    per = [(sum(lut[(v, w, c['d_model'])]['mc'] for v, w in embedding_tables(c)), bottleneck_embedding_capacity(c))
           for c in configs.values()]
    pr = np.array([a / b - 1 for a, b in per])
    a, b = np.array(per).T
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    print(f'   per configuration (200): rel err mean {pr.mean():+.3f}, max |.| {np.abs(pr).max():.3f}; '
          f'Spearman(MC path, cap) = {np.corrcoef(ra, rb)[0, 1]:.4f}')

    print('\n2. capped embedding capacity by width and d_embed (mean over all 200 configs, no labels)')
    for w in (64, 128, 256, 512):
        line = []
        for de in (128, 256, 512):
            vals = [bottleneck_embedding_capacity(c) for c in configs.values() if c['d_model'] == w and c['d_embed'] == de]
            divs = {}
            for c in configs.values():
                if c['d_model'] == w and c['d_embed'] == de:
                    divs.setdefault(c['div_val'], bottleneck_embedding_capacity(c))
            line.append(f"de={de}: {np.mean(vals):7.1f} (div1/2/4: " + '/'.join(f'{divs.get(k, float("nan")):.0f}' for k in (1, 2, 4)) + ')')
        print(f'   d={w:3d}  ' + '  '.join(line))

    print('\n3. controlled pair: same layers (d=256, L=6, H=4, f=1024), div_val=1, d_embed varied')
    base = dict(d_model=256, n_layer=6, n_head=4, d_inner=1024, n_token=50264, div_val=1,
                cutoffs=[19997, 39997, 199997, 50264], tie_projs=[False, True, True, True])
    for de in (128, 256, 512):
        c = dict(base, d_embed=de)
        print(f'   d_embed={de}: NSC={original_nsc(c):8.1f}  V1={original_nsc(c) + bottleneck_embedding_capacity(c):8.1f}'
              f'  uncapped={original_nsc(c) + embedding_capacity(c):8.1f}  proj params={resource_estimate(c)["projection_parameters"]}')

    (Path(__file__).with_name('mechanism_check.json')).write_text(json.dumps(rows, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

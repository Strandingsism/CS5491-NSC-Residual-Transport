"""Residual transport factors: spec-only prediction, Monte Carlo check, label-free ranking impact.

Reference model (stated assumptions, not the exact archived Transformer-XL implementation):
- Sublayers in order attn_1, ffn_1, ..., attn_L, ffn_L, each with branch output u_s = F_s(.).
    post: x <- LN(x + F(x))          (archive: pre_lnorm=false)
    pre:  x <- x + F(LN(x)), final LN
    none: x <- x + F(x)
- Causal softmax attention with per-head Q/K/V and one output projection; FFN W2 relu(W1 x).
  All entries N(0, s^2), s=0.02 (NSC convention); zero biases; LN gain 1, bias 0; no dropout.
  No memory segment, relative-position terms, embeddings or vocabulary head.
- Inputs: n iid tokens with N(0,1) entries (normalized token statistics).

Transport T_s = ||dy/du_s||_F^2 / (n d): mean squared gain from branch output s to the output y.
Weighted score: sum_s T_s * Psi_s with Psi_s the starter's MP capacity of sublayer s's matrices.
Setting every T_s = 1 recovers original NSC (the ablation).

Run from the project root:  .venv/Scripts/python.exe analysis/postln_transport/transport.py
No label file is read. Outputs are written beside this file.
"""
from pathlib import Path
import json
import math
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
STARTER = OUT.parents[1] / 'nsc-starter'
sys.path.insert(0, str(STARTER))
from nsc import layers, psi_mp, original_nsc, position_example  # noqa: E402
from evaluate import pair_summary  # noqa: E402  (pure function; reads no labels)

SIGMA = 0.02
NORMS = ('post', 'pre', 'none')


# ---------------------------------------------------------------- spec-only prediction

def attention_key_factor(n_query, mem=0):
    """Mean over queries of 1/(keys seen): variance kept by uniform causal averaging of iid tokens."""
    return sum(1.0 / (mem + i) for i in range(1, n_query + 1)) / n_query


def branch_gains(d, heads, inner, sigma=SIGMA, key_factor=1.0):
    """Per-sublayer gain v_s = output variance / input variance (= Jacobian gain for iid inputs).

    Attention at small init is near-uniform causal averaging: v = d^2 s^4 * key_factor.
    FFN with ReLU on zero-mean Gaussian input: v = f d s^4 / 2.
    """
    gains = []
    for f in inner:
        gains.append(d * d * sigma ** 4 * key_factor)
        gains.append(f * d * sigma ** 4 / 2)
    return gains


def predict_transport(gains, d, norm):
    """Predicted T_s for each sublayer under one normalization placement."""
    S = len(gains)
    if norm == 'none':
        return [math.prod(1 + g for g in gains[s + 1:]) for s in range(S)]
    keep = 1 - 2 / d  # each LN Jacobian projects out the mean and the normalized-output direction
    if norm == 'post':
        # Own LN divides by 1+v_s; each later sublayer adds gain (1+v_t) and its LN divides it back.
        return [keep / (1 + gains[s]) for s in range(S)]
    # pre: stream variance grows; branch sees LN(x), so its residual Jacobian gain is v_t / var_t.
    var = [1.0]
    for g in gains:
        var.append(var[-1] + g)
    return [keep * math.prod(1 + gains[t] / var[t] for t in range(s + 1, S)) / var[-1]
            for s in range(S)]


def sublayer_capacities(config):
    """Starter's per-layer capacity split into (attention, FFN) sublayers; sums to original NSC."""
    d, heads, inner = layers(config)
    caps = []
    for h, f in zip(heads, inner):
        caps.append(3 * h * psi_mp(d, d // h) + psi_mp(d, d))
        caps.append(2 * psi_mp(d, f))
    return caps


def archive_key_factor(config):
    return attention_key_factor(config.get('tgt_len', 192), config.get('mem_len', 192))


def transport_score(config, norm='post'):
    """Specification-only transport-weighted NSC."""
    d, heads, inner = layers(config)
    T = predict_transport(branch_gains(d, heads, inner, key_factor=archive_key_factor(config)), d, norm)
    return math.fsum(t * c for t, c in zip(T, sublayer_capacities(config)))


# ---------------------------------------------------------------- Monte Carlo reference

def layer_norm(z):
    c = z - z.mean(-1, keepdims=True)
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + 1e-5)


def make_sublayers(d, heads, inner, rng, sigma=SIGMA):
    subs = []
    for h, f in zip(heads, inner):
        wq, wk, wv, wo = (rng.normal(0, sigma, (d, d)) for _ in range(4))
        w1, w2 = rng.normal(0, sigma, (d, f)), rng.normal(0, sigma, (f, d))

        def attn(x, wq=wq, wk=wk, wv=wv, wo=wo, h=h):
            n, dh = x.shape[0], d // h
            q, k, v = ((x @ w).reshape(n, h, dh).transpose(1, 0, 2) for w in (wq, wk, wv))
            s = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
            s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
            a = np.exp(s - s.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            return (a @ v).transpose(1, 0, 2).reshape(n, d) @ wo

        def ffn(x, w1=w1, w2=w2):
            return np.maximum(x @ w1, 0) @ w2
        subs += [attn, ffn]
    return subs


def run(x, subs, norm, start=0, delta=None, record=None):
    for i in range(start, len(subs)):
        if record is not None:
            record.append(x)
        inp = layer_norm(x) if norm == 'pre' else x
        u = subs[i](inp)
        if record is not None:
            record[-1] = (x, float((u * u).mean() / (inp * inp).mean()))
        if delta is not None and i == start:
            u = u + delta
        x = layer_norm(x + u) if norm == 'post' else x + u
    return layer_norm(x) if norm == 'pre' else x


def measure_transport(d, heads, inner, norm, n_tokens=64, seeds=2, probes=2, eps=1e-4):
    """Hutchinson estimate of T_s by central finite differences; also measured gains v_s."""
    S = 2 * len(inner)
    T, V = np.zeros(S), np.zeros(S)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        subs = make_sublayers(d, heads, inner, rng)
        x0 = rng.normal(size=(n_tokens, d))
        rec = []
        run(x0, subs, norm, record=rec)
        for s in range(S):
            V[s] += rec[s][1]
            for _ in range(probes):
                v = rng.normal(size=(n_tokens, d))
                diff = run(rec[s][0], subs, norm, s, eps * v) - run(rec[s][0], subs, norm, s, -eps * v)
                T[s] += (diff * diff).sum() / (4 * eps * eps) / (n_tokens * d) / probes
    return T / seeds, V / seeds


# ---------------------------------------------------------------- experiments

def mc_validation(configs, keys, n_tokens=64):
    """Compare predicted and measured T_s on real development-split specifications."""
    kf = attention_key_factor(n_tokens)
    rows = []
    for key in keys:
        d, heads, inner = layers(configs[key])
        gains = branch_gains(d, heads, inner, key_factor=kf)
        for norm in NORMS:
            T, V = measure_transport(d, heads, inner, norm, n_tokens)
            P = predict_transport(gains, d, norm)
            rows.append(dict(key=key, d=d, L=len(inner), norm=norm,
                             measured=T.tolist(), predicted=P, measured_gain=V.tolist(), predicted_gain=gains))
    return rows


def summarize_mc(rows):
    out = {}
    for norm in NORMS:
        sel = [r for r in rows if r['norm'] == norm]
        rel = np.concatenate([np.array(r['predicted']) / np.array(r['measured']) - 1 for r in sel])
        spread = [max(r['measured']) / min(r['measured']) for r in sel]
        # Within-config position trend of measured T (Spearman-like via rank correlation with index).
        trend = []
        for r in sel:
            m = np.array(r['measured'])
            if len(m) > 2 and m.std() > 0:
                trend.append(float(np.corrcoef(np.argsort(np.argsort(m)), np.arange(len(m)))[0, 1]))
        out[norm] = dict(configs=len(sel),
                         rel_error_mean=float(rel.mean()), rel_error_max_abs=float(np.abs(rel).max()),
                         within_config_max_over_min_median=float(np.median(spread)),
                         within_config_max_over_min_max=float(max(spread)),
                         position_rank_trend_mean=float(np.mean(trend)) if trend else None)
    return out


def ranking_impact(configs, keys, totals):
    """Label-free: how far does each weighting move the development ranking away from original NSC?"""
    methods = {'post_spec': lambda c: transport_score(c, 'post'),
               'pre_spec_mismatch': lambda c: transport_score(c, 'pre'),
               'none_spec': lambda c: transport_score(c, 'none'),
               'position_example': position_example}
    nsc = [original_nsc(configs[k]) for k in keys]
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if configs[keys[i]]['d_model'] == configs[keys[j]]['d_model']
               and min(totals[i], totals[j]) / max(totals[i], totals[j]) >= 0.90]
    result = {}
    for name, fn in methods.items():
        s = [fn(configs[k]) for k in keys]
        topk = {}
        for cap in (10_000_000, 20_000_000, 30_000_000, 50_000_000):
            pool = [i for i, t in enumerate(totals) if t <= cap]
            for k in (1, 3):
                a = set(sorted(pool, key=lambda i: -nsc[i])[:k])
                b = set(sorted(pool, key=lambda i: -s[i])[:k])
                topk[f'{cap // 1_000_000}M_k{k}'] = len(a ^ b) // 2
        allp, mp = pair_summary(s, nsc, pairs), pair_summary(s, nsc, matched)
        result[name] = dict(tau_b_vs_nsc=allp['tau_b'], flipped_pairs_vs_nsc=allp['discordant'],
                            all_pairs=allp['pairs'], matched_pairs=mp['pairs'],
                            matched_flipped_vs_nsc=mp['discordant'], topk_changed_members=topk)
    return result


def weight_profiles(configs, keys):
    """Spread of spec-only Post-LN weights on the panel."""
    spreads, lo, hi = [], 1.0, 0.0
    for k in keys:
        d, heads, inner = layers(configs[k])
        T = predict_transport(branch_gains(d, heads, inner, key_factor=archive_key_factor(configs[k])), d, 'post')
        spreads.append(max(T) / min(T))
        lo, hi = min(lo, min(T)), max(hi, max(T))
    return dict(min_T=lo, max_T=hi, within_config_max_over_min_median=float(np.median(spreads)),
                within_config_max_over_min_max=float(max(spreads)))


def main():
    start = time.perf_counter()
    data = STARTER / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    keys = json.loads((data / 'splits.json').read_text())['development']
    resources = json.loads((data / 'resources.json').read_text())
    totals = [resources[k]['total'] for k in keys]

    toy = {'d_model': 128, 'n_layer': 2, 'n_head': [4, 4], 'd_inner': [256, 512]}
    rev = dict(toy, d_inner=[512, 256])
    toy_result = {n: [transport_score(toy, n), transport_score(rev, n)] for n in NORMS}

    # Monte Carlo on a width/depth-stratified subset of development specifications (labels unused).
    by_width = {}
    for k in keys:
        by_width.setdefault(configs[k]['d_model'], []).append(k)
    subset = []
    for w in sorted(by_width):
        ks = sorted(by_width[w], key=lambda k: configs[k]['n_layer'])
        subset += [ks[0], ks[len(ks) // 2], ks[-1]]
    mc_rows = mc_validation(configs, subset)

    result = dict(
        assumptions=__doc__.split('\n\n')[1],
        toy_reorder={n: dict(ffn_256_512=v[0], ffn_512_256=v[1]) for n, v in toy_result.items()},
        mc_subset=subset, mc_summary=summarize_mc(mc_rows),
        post_weight_profile=weight_profiles(configs, keys),
        ranking_impact_vs_nsc=ranking_impact(configs, keys, totals),
        seconds=time.perf_counter() - start)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (OUT / 'mc_rows.json').write_text(json.dumps(mc_rows, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'assumptions'}, indent=2))


if __name__ == '__main__':
    main()

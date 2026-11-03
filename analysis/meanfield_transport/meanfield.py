"""Mean-field residual transport with token correlation and detached memory.

Refines analysis/postln_transport (v1), whose iid-token assumption failed for wide,
deep Post-LN models: causal averaging and ReLU means make tokens correlated layer by
layer (the rank-collapse mechanism of Dong et al. 2021; Noci et al. 2022), so the
attention forward gain v exceeds its Jacobian gain g and (1+g)/(1+v) < 1 compounds.

Reference model (same as v1 unless stated):
- Sublayers attn_1, ffn_1, ..., attn_L, ffn_L; post x<-LN(x+F(x)), pre x<-x+F(LN(x)) + final LN,
  none x<-x+F(x). All matrix entries N(0, s^2), s=0.02; zero biases; LN gain 1; no dropout.
- Attention at small init is uniform causal averaging over K_i = mem + i keys, then W_V W_O.
  Memory states (Transformer-XL) enter the forward pass but are detached (stop-gradient).
- Input tokens iid N(0,1): pairwise token correlation rho_0 = 0.

State per sublayer input: per-coordinate energy E and mean pairwise inner product P = rho E.
Forward branch output (energy m, pairwise p) for unit-energy input with correlation rho:
  attn: m = a (c1 + rho (1-c1)),          p = a (c2 + rho (1-c2)),           a = d^2 s^4
  ffn:  m = b,                            p = 2 b k1(rho),                    b = f d s^4 / 2
  c1 = mean_i 1/K_i, c2 = mean_{i<k} 1/K_k, k1 = arc-cosine kernel of degree 1 (normalized).
Backward: a perturbation injected at branch s (iid, energy 1, pairwise 0) propagates as (n, q):
  attn: n += a (A1 n + A2 q),  q += a (B1 n + B2 q)   (only current tokens carry perturbation)
  ffn:  n += b n,              q += 2 b k0(rho) q     (k0 = P(both ReLU units active))
  with each LayerNorm dividing by the stream energy and removing ~2 directions once (1-2/d).
T_s = final n (mean squared output gain per token and coordinate), as in v1.

Run from project root: .venv/Scripts/python.exe analysis/meanfield_transport/meanfield.py
No label file is read.
"""
from functools import lru_cache
from pathlib import Path
import json
import math
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
sys.path.insert(0, str(ROOT / 'analysis' / 'postln_transport'))
from nsc import layers  # noqa: E402
import transport as v1  # noqa: E402

SIGMA = 0.02
NORMS = ('post', 'pre', 'none')


# ---------------------------------------------------------------- spec-only mean-field prediction

def k0(rho):
    return (math.pi - math.acos(max(-1.0, min(1.0, rho)))) / (2 * math.pi)


def k1(rho):
    rho = max(-1.0, min(1.0, rho))
    return (math.sqrt(1 - rho * rho) + (math.pi - math.acos(rho)) * rho) / (2 * math.pi)


@lru_cache(maxsize=None)
def causal_coefficients(n, mem):
    """Uniform-averaging coefficients for n current queries and mem detached memory keys."""
    i = np.arange(1, n + 1, dtype=float)
    K = mem + i
    c1 = float(np.mean(1 / K))
    A1 = float(np.mean(i / K ** 2))
    A2 = float(np.mean(i * (i - 1) / K ** 2))
    if n < 2:
        return c1, c1, A1, A2, 0.0, 0.0
    # pairs i<k (1-based): forward overlap K_i keys, backward overlap i current keys
    ii, kk = np.triu_indices(n, 1)
    Ki, Kk, iv, kv = K[ii], K[kk], i[ii], i[kk]
    c2 = float(np.mean(1 / Kk))
    B1 = float(np.mean(iv / (Ki * Kk)))
    B2 = float(np.mean(iv * (kv - 1) / (Ki * Kk)))
    return c1, c2, A1, A2, B1, B2


def sublayer_kinds(d, inner, sigma=SIGMA):
    out = []
    for f in inner:
        out.append(('attn', d * d * sigma ** 4))
        out.append(('ffn', f * d * sigma ** 4 / 2))
    return out


def forward_state(d, inner, norm, n, mem, sigma=SIGMA):
    """Per sublayer: input energy E, input correlation rho, branch output energy m (for that input)."""
    c1, c2, *_ = causal_coefficients(n, mem)
    E, rho, rows = 1.0, 0.0, []
    for kind, g in sublayer_kinds(d, inner, sigma):
        scale = E if norm == 'none' else 1.0  # pre/post branches see unit-energy input
        if kind == 'attn':
            m, p = g * (c1 + rho * (1 - c1)), g * (c2 + rho * (1 - c2))
        else:
            m, p = g, 2 * g * k1(rho)
        m, p = m * scale, p * scale
        rows.append((E, rho, m))
        E_new, P_new = E + m, rho * E + p
        rho = P_new / E_new
        E = 1.0 if norm == 'post' else E_new
    return rows, E


def predict_meanfield(d, inner, norm='post', n=192, mem=192, sigma=SIGMA):
    """Spec-only transport T_s for every sublayer (v2)."""
    kinds = sublayer_kinds(d, inner, sigma)
    rows, E_final = forward_state(d, inner, norm, n, mem, sigma)
    _, _, A1, A2, B1, B2 = causal_coefficients(n, mem)
    keep = 1 - 2 / d
    S, T = len(kinds), []
    for s in range(S):
        nn, q = 1.0, 0.0
        if norm == 'post':
            nn /= rows[s][0] + rows[s][2]
        for t in range(s + 1, S):
            kind, g = kinds[t]
            E_t, rho_t, m_t = rows[t]
            # branch Jacobian gain on the stream perturbation; pre sees LN(x): divide by E_t
            scale = 1.0 / E_t if norm == 'pre' else 1.0
            if kind == 'attn':
                dn, dq = g * (A1 * nn + A2 * q), g * (B1 * nn + B2 * q)
            else:
                dn, dq = g * nn, 2 * g * k0(rho_t) * q
            nn, q = nn + scale * dn, q + scale * dq
            if norm == 'post':
                nn, q = nn / (E_t + m_t), q / (E_t + m_t)
        if norm == 'pre':
            nn, q = nn / E_final, q / E_final
        T.append(keep * nn if norm != 'none' else nn)
    return T


def predicted_correlation(d, inner, norm, n, mem):
    rows, _ = forward_state(d, inner, norm, n, mem)
    return [r[1] for r in rows]


# ---------------------------------------------------------------- Monte Carlo reference (with memory)

def layer_norm(z):
    c = z - z.mean(-1, keepdims=True)
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + 1e-5)


def make_sublayers(d, heads, inner, rng, sigma=SIGMA):
    subs = []
    for h, f in zip(heads, inner):
        wq, wk, wv, wo = (rng.normal(0, sigma, (d, d)) for _ in range(4))
        w1, w2 = rng.normal(0, sigma, (d, f)), rng.normal(0, sigma, (f, d))

        def attn(x, mem, wq=wq, wk=wk, wv=wv, wo=wo, h=h):
            n, dh, m = x.shape[0], d // h, mem.shape[0]
            kv_in = np.concatenate([mem, x]) if m else x
            q = (x @ wq).reshape(n, h, dh).transpose(1, 0, 2)
            k, v = ((kv_in @ w).reshape(m + n, h, dh).transpose(1, 0, 2) for w in (wk, wv))
            s = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
            mask = np.tril(np.ones((n, m + n), bool), m)
            s = np.where(mask, s, -np.inf)
            a = np.exp(s - s.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            return (a @ v).transpose(1, 0, 2).reshape(n, d) @ wo

        def ffn(x, mem, w1=w1, w2=w2):
            return np.maximum(x @ w1, 0) @ w2
        subs += [attn, ffn]
    return subs


def run(x, subs, norm, mems, start=0, delta=None, record=None):
    """mems[i]: memory for sublayer i (states entering that sublayer in the previous segment)."""
    for i in range(start, len(subs)):
        if record is not None:
            record.append(x)
        inp = layer_norm(x) if norm == 'pre' else x
        mem = mems[i] if mems is not None else np.zeros((0, x.shape[1]))
        mem_in = layer_norm(mem) if (norm == 'pre' and mem.shape[0]) else mem
        u = subs[i](inp, mem_in)
        if delta is not None and i == start:
            u = u + delta
        x = layer_norm(x + u) if norm == 'post' else x + u
    return layer_norm(x) if norm == 'pre' else x


def mean_pairwise_cos(x):
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    g = u @ u.T
    n = len(x)
    return float((g.sum() - n) / (n * (n - 1)))


def measure(d, heads, inner, norm, n=64, mem=64, seeds=2, probes=2, eps=1e-4, warmup=2):
    S = 2 * len(inner)
    T, R = np.zeros(S), np.zeros(S)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        subs = make_sublayers(d, heads, inner, rng)
        mems = None
        if mem:
            for _ in range(warmup):  # previous segments produce (approximately stationary) memory
                rec = []
                run(rng.normal(size=(mem, d)), subs, norm, mems, record=rec)
                mems = rec
        x0 = rng.normal(size=(n, d))
        rec = []
        run(x0, subs, norm, mems, record=rec)
        for s in range(S):
            R[s] += mean_pairwise_cos(rec[s])
            for _ in range(probes):
                v = rng.normal(size=(n, d))
                diff = run(rec[s], subs, norm, mems, s, eps * v) - run(rec[s], subs, norm, mems, s, -eps * v)
                T[s] += (diff * diff).sum() / (4 * eps * eps) / (n * d) / probes
    return T / seeds, R / seeds


# ---------------------------------------------------------------- experiments

def validation(configs, keys, settings):
    rows = []
    for n, mem in settings:
        kf = v1.attention_key_factor(n, mem)
        for key in keys:
            d, heads, inner = layers(configs[key])
            for norm in NORMS:
                T, R = measure(d, heads, inner, norm, n, mem)
                rows.append(dict(key=key, d=d, L=len(inner), norm=norm, n=n, mem=mem,
                                 measured=T.tolist(), measured_rho=R.tolist(),
                                 v1=v1.predict_transport(v1.branch_gains(d, heads, inner, key_factor=kf), d, norm),
                                 v2=predict_meanfield(d, inner, norm, n, mem),
                                 v2_rho=predicted_correlation(d, inner, norm, n, mem)))
    return rows


def summarize(rows):
    out = {}
    for n, mem in sorted({(r['n'], r['mem']) for r in rows}):
        for norm in NORMS:
            sel = [r for r in rows if r['norm'] == norm and r['n'] == n and r['mem'] == mem]
            entry = {}
            for model in ('v1', 'v2'):
                rel = np.concatenate([np.array(r[model]) / np.array(r['measured']) - 1 for r in sel])
                # within-config shape: correlation of predicted vs measured T profile
                shape = [float(np.corrcoef(r[model], r['measured'])[0, 1]) for r in sel
                         if len(r['measured']) > 4 and np.std(r[model]) > 0]
                entry[model] = dict(rel_error_mean=float(rel.mean()), rel_error_rms=float(np.sqrt((rel ** 2).mean())),
                                    rel_error_max_abs=float(np.abs(rel).max()),
                                    profile_corr_mean=float(np.mean(shape)) if shape else None)
            rho_err = np.concatenate([np.array(r['v2_rho']) - np.array(r['measured_rho']) for r in sel])
            entry['rho_abs_error_mean'] = float(np.abs(rho_err).mean())
            entry['rho_measured_last_mean'] = float(np.mean([r['measured_rho'][-1] for r in sel]))
            entry['rho_v2_last_mean'] = float(np.mean([r['v2_rho'][-1] for r in sel]))
            entry['measured_max_over_min_median'] = float(np.median([max(r['measured']) / min(r['measured']) for r in sel]))
            out[f'n{n}_mem{mem}_{norm}'] = entry
    return out


def main():
    start = time.perf_counter()
    data = ROOT / 'nsc-starter' / 'data'
    configs = json.loads((data / 'architectures.json').read_text())
    subset = json.loads((ROOT / 'analysis' / 'postln_transport' / 'results.json').read_text())['mc_subset']
    rows = validation(configs, subset, settings=((64, 0), (64, 64)))
    result = dict(subset=subset, summary=summarize(rows), seconds=time.perf_counter() - start)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (OUT / 'validation_rows.json').write_text(json.dumps(rows, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=1))
    print(f"{result['seconds']:.0f}s")


if __name__ == '__main__':
    main()

"""Specification-only residual transport factors for NSC layer weighting.

Standard library only. Derivation, assumptions and Monte Carlo validation:
analysis/meanfield_transport/ (numpy reference implementation of the same recursion).

T_s is the mean squared gain from the output of residual branch s (attn_1, ffn_1, ...,
attn_L, ffn_L) to the network output at initialization, under a mean-field model that
tracks per-token energy and mean pairwise token correlation:

- all matrix entries N(0, s^2), s=0.02; zero biases; LN gain 1; no dropout;
- attention at small init is uniform causal averaging over mem+i keys, then W_V W_O;
  memory is used in the forward pass and detached in the backward pass (Transformer-XL);
- FFN W2 relu(W1 x); ReLU correlation maps are the arc-cosine kernels k0, k1;
- input tokens iid with zero pairwise correlation.

Placements: post x<-LN(x+F(x)) (the archive), pre x<-x+F(LN(x)) with final LN, none x<-x+F(x).
"""
from functools import lru_cache
import math

SIGMA = 0.02


def _k0(rho):
    return (math.pi - math.acos(max(-1.0, min(1.0, rho)))) / (2 * math.pi)


def _k1(rho):
    rho = max(-1.0, min(1.0, rho))
    return (math.sqrt(1 - rho * rho) + (math.pi - math.acos(rho)) * rho) / (2 * math.pi)


@lru_cache(maxsize=None)
def causal_coefficients(n, mem):
    """Averages over n causal queries with mem detached memory keys; K_i = mem + i.

    c1 = mean_i 1/K_i, c2 = mean_{i<k} 1/K_k (forward); A1, A2 (energy) and B1, B2
    (pairwise) weight a perturbation that lives on the i current keys only (backward).
    """
    if n < 1 or mem < 0:
        raise ValueError('need n >= 1 and mem >= 0')
    K = [mem + i for i in range(1, n + 1)]
    c1 = math.fsum(1 / k for k in K) / n
    A1 = math.fsum(i / K[i - 1] ** 2 for i in range(1, n + 1)) / n
    A2 = math.fsum(i * (i - 1) / K[i - 1] ** 2 for i in range(1, n + 1)) / n
    if n < 2:
        return c1, c1, A1, A2, 0.0, 0.0
    pairs = n * (n - 1) / 2
    c2 = B1 = B2 = 0.0
    prefix = 0.0  # sum_{i<k} i / K_i
    for k in range(1, n + 1):
        Kk = K[k - 1]
        c2 += (k - 1) / Kk
        B1 += prefix / Kk
        B2 += prefix * (k - 1) / Kk
        prefix += k / Kk
    return c1, c2 / pairs, A1, A2, B1 / pairs, B2 / pairs


def _sublayers(d, inner, sigma):
    out = []
    for f in inner:
        out.append(('attn', d * d * sigma ** 4))
        out.append(('ffn', f * d * sigma ** 4 / 2))
    return out


def transport_factors(d, inner, norm='post', n=192, mem=192, sigma=SIGMA):
    """Predicted T_s for every sublayer; T_s -> 1 - 2/d (LN) or 1 (none) as sigma -> 0."""
    if norm not in ('post', 'pre', 'none'):
        raise ValueError('norm must be post, pre or none')
    kinds = _sublayers(d, inner, sigma)
    c1, c2, A1, A2, B1, B2 = causal_coefficients(n, mem)
    # Forward: input energy E, correlation rho and branch output energy m at each sublayer.
    E, rho, rows = 1.0, 0.0, []
    for kind, g in kinds:
        scale = E if norm == 'none' else 1.0
        if kind == 'attn':
            m, p = g * (c1 + rho * (1 - c1)), g * (c2 + rho * (1 - c2))
        else:
            m, p = g, 2 * g * _k1(rho)
        m, p = m * scale, p * scale
        rows.append((E, rho, m))
        rho = (rho * E + p) / (E + m)
        E = 1.0 if norm == 'post' else E + m
    # Backward: perturbation energy nn and pairwise inner product q, injected iid at branch s.
    keep = 1.0 if norm == 'none' else 1 - 2 / d
    T = []
    for s in range(len(kinds)):
        nn, q = 1.0, 0.0
        if norm == 'post':
            nn /= rows[s][0] + rows[s][2]
        for t in range(s + 1, len(kinds)):
            kind, g = kinds[t]
            E_t, rho_t, m_t = rows[t]
            scale = 1.0 / E_t if norm == 'pre' else 1.0
            if kind == 'attn':
                dn, dq = g * (A1 * nn + A2 * q), g * (B1 * nn + B2 * q)
            else:
                dn, dq = g * nn, 2 * g * _k0(rho_t) * q
            nn, q = nn + scale * dn, q + scale * dq
            if norm == 'post':
                nn, q = nn / (E_t + m_t), q / (E_t + m_t)
        if norm == 'pre':
            nn /= E
        T.append(keep * nn)
    return T

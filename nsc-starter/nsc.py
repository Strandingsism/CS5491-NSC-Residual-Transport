"""Specification-only NSC and explicit, simplified resource estimates.

Python 3.9+, standard library only. The baseline reproduces the NSC paper's
per-head Q/K/V convention for this archived panel, with fixed sigma=0.02.
It excludes embeddings, biases, normalization and the language-model head.
"""
from functools import lru_cache
import math


@lru_cache(maxsize=None)
def psi_mp(m, n, sigma=0.02, steps=512):
    """MP approximation to E[ln det(I+W.T W)] for iid N(0,sigma^2) W.

    Substitute x=(1-sqrt(gamma))^2+4sqrt(gamma)sin^2(theta/2).
    MP density times dx becomes 2 sin^2(theta)/(pi*x) dtheta.
    Composite Simpson integration avoids the density's endpoint singularity.
    This is an asymptotic spectral approximation, not an exact finite-size mean.
    """
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (m, n)):
        raise ValueError("Matrix dimensions must be positive integers")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not isinstance(steps, int) or steps < 2 or steps % 2:
        raise ValueError("steps must be a positive even integer >= 2")
    m, n = max(m, n), min(m, n)
    root = math.sqrt(n / m)
    a = sigma * sigma * m
    h = math.pi / steps
    terms = []
    for j in range(1, steps):  # Both endpoints vanish, including gamma=1.
        theta = j * h
        x = (1 - root) ** 2 + 4 * root * math.sin(theta / 2) ** 2
        f = math.sin(theta) ** 2 * math.log1p(a * x) / x
        terms.append((4 if j % 2 else 2) * f)
    return n * 2 / math.pi * h / 3 * math.fsum(terms)


def layers(config):
    """Return (hidden width, head counts, FFN widths) after schema checks."""
    d, length = config['d_model'], config['n_layer']
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (d, length)):
        raise ValueError("d_model and n_layer must be positive integers")
    heads, inner = config['n_head'], config['d_inner']
    heads = [heads] * length if isinstance(heads, int) else list(heads)
    inner = [inner] * length if isinstance(inner, int) else list(inner)
    if len(heads) != length or len(inner) != length:
        raise ValueError("Per-layer list lengths must equal n_layer")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in heads + inner):
        raise ValueError("Head counts and FFN widths must be positive integers")
    if any(d % h for h in heads):
        raise ValueError("d_model must be divisible by each head count")
    return d, heads, inner


def layer_capacities(config):
    """Each layer: 3 H psi(d,d/H) + psi(d,d) + 2 psi(d,f)."""
    d, heads, inner = layers(config)
    return [3 * h * psi_mp(d, d // h) + psi_mp(d, d) + 2 * psi_mp(d, f)
            for h, f in zip(heads, inner)]


def original_nsc(config):
    """Higher means greater additive reference capacity, not guaranteed quality."""
    return math.fsum(layer_capacities(config))


def position_example(config):
    """Teaching example: increasing layer weights; not a validated capacity model.

    This arbitrary positional weighting demonstrates how to break a reorder
    tie. It is not a validated graph model or a sufficient project contribution.
    """
    values = layer_capacities(config)
    return math.fsum((i + 1) / len(values) * v for i, v in enumerate(values))


def resource_estimate(config, key_length=384):
    """Count projection weights and dense attention MACs per query token.

    P_proj=sum_l(4*d^2+2*d*f_l). MACs/token=P_proj+2*L*key_length*d.
    The second term counts QK and AV. FLOPs=2*MACs. Fixed key length 384
    corresponds to a reference 192-token segment plus 192 cached states.
    This counts full dense products, not triangular/sparse kernel savings.
    Omits embeddings, output vocabulary head, bias, norm, softmax, activation
    and cache preparation. It is neither total model FLOPs nor measured latency.
    """
    if not isinstance(key_length, int) or key_length < 1:
        raise ValueError("key_length must be a positive integer")
    d, heads, inner = layers(config)
    projection = sum(4 * d * d + 2 * d * f for f in inner)
    macs = projection + 2 * len(heads) * key_length * d
    return {'projection_parameters': projection,
            'reference_macs_per_token': macs,
            'reference_flops_per_token': 2 * macs}

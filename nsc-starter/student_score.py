"""Residual-transport-weighted NSC (Post-LN), and its ablation.

Mechanism: each sublayer's MP capacity is weighted by how strongly its branch output is
transported to the network output at initialization, T_s from residual_transport.py.
The weights come from the specification only (widths, depth, sequence/memory lengths,
sigma=0.02, the archive's pre_lnorm=false); no label, ID or fitted constant is used.

Ablation: T_s = 1 for every sublayer, which is exactly original NSC.

Frozen development choices (made from label-free Monte Carlo validation only):
- placement 'post' because every archived configuration has pre_lnorm=false;
- mean-field recursion with token correlation (v2), not the iid-token version (v1);
- n = tgt_len, mem = mem_len from the configuration (192/192 for the whole panel).
"""
import math

from nsc import layers, psi_mp, original_nsc
from residual_transport import transport_factors


def sublayer_capacities(config):
    """Starter's per-layer capacity split into (attention, FFN); sums to original NSC."""
    d, heads, inner = layers(config)
    caps = []
    for h, f in zip(heads, inner):
        caps.append(3 * h * psi_mp(d, d // h) + psi_mp(d, d))
        caps.append(2 * psi_mp(d, f))
    return caps


def transport_weighted_nsc(config, norm='post'):
    d, _, inner = layers(config)
    T = transport_factors(d, inner, norm, config.get('tgt_len', 192), config.get('mem_len', 192))
    return math.fsum(t * c for t, c in zip(T, sublayer_capacities(config)))


def score(config):
    """Return a finite scalar, with larger values predicting better quality."""
    return transport_weighted_nsc(config, 'post')


def ablation(config):
    """Removes the transport weights (T_s = 1): original NSC."""
    return original_nsc(config)

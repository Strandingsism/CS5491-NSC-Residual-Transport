"""Specification-only capacity of the tied adaptive-embedding path (v3). Standard library only.

Every token reaches the residual stream through its cluster's embedding table E (v tokens x w
columns) followed by a w x d_model projection P; the same tables are reused by the tied output
softmax. NSC scores neither. Under Post-LN the overall scale of P is removed (see v2,
residual_transport.py), so P is taken variance-preserving, N(0, 1/w). Because v >> w in this panel
(w/v <= 0.05), E^T E ~= sigma^2 v I_w and the nonzero spectrum of E P is that of a w x d matrix with
entries N(0, sigma^2 v / w). Its capacity is therefore psi_MP(w, d, sigma * sqrt(v / w)), which
matches Monte Carlo within 0.4% RMS over all 60 table shapes in the panel
(analysis/v1_explore/mechanism_check.py). A div_val=1 table with d_embed == d_model has no
projection and is scored as psi_MP(v, d, sigma). Each tied table is counted once.

Table shapes follow the archive's Transformer-XL adaptive embedding: cutoffs truncated at n_token,
cluster i has width d_embed / div_val**i (a single n_token x d_embed table when div_val == 1).
Together with 192 * d_model position-related parameters these shapes reproduce resources.json's
AdaEmb exactly for all 200 configurations.
"""
import math

from nsc import psi_mp

SIGMA = 0.02


def embedding_tables(config):
    """(tokens, width) of each adaptive-embedding table."""
    n, de, div = config['n_token'], config['d_embed'], config['div_val']
    if div == 1:
        return [(n, de)]
    ends = sorted({min(x, n) for x in config['cutoffs']})
    return [(b - a, de // div ** i) for i, (a, b) in enumerate(zip([0] + ends[:-1], ends))]


def path_embedding_capacity(config, sigma=SIGMA):
    """Sum over tables of the MP capacity of the embedding path E P into the residual stream."""
    d = config['d_model']
    total = []
    for v, w in embedding_tables(config):
        if config['div_val'] == 1 and w == d:
            total.append(psi_mp(v, d, sigma))
        else:
            total.append(psi_mp(w, d, sigma * math.sqrt(v / w)))
    return math.fsum(total)

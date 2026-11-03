"""v3: NSC plus the capacity of the embedding path, and its ablation.

Mechanism: NSC sums the MP capacity of the attention and FFN matrices but ignores the tied adaptive
embedding, through which every token enters and leaves the network. v3 adds the capacity of each
embedding table seen through its rank-limited projection into the residual stream
(embedding_path.py). Specification only: no label, ID or fitted constant.

Ablation: remove the embedding path, which is exactly original NSC.

Frozen development choices (analysis/v3_embedding_path/README.md, sections 1-7):
- additive aggregation (the series alternative failed as pre-registered, section 9);
- path capacity with a variance-preserving projection (option B), chosen from label-free
  Monte Carlo checks before its development evaluation;
- sigma = 0.02, NSC's convention; each tied table counted once.

Evaluate with: python nsc-starter/evaluate.py --split development --student nsc-starter/v3_score.py
"""
from nsc import original_nsc
from embedding_path import path_embedding_capacity


def score(config):
    """Return a finite scalar, with larger values predicting better quality."""
    return original_nsc(config) + path_embedding_capacity(config)


def ablation(config):
    """Removes the embedding path: original NSC."""
    return original_nsc(config)

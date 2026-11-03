"""Transformer-XL with tied adaptive embeddings, following the archived panel's conventions.

What matches the archive (nsc-starter/data/architectures.json): relative-position attention with
segment memory (attn_type 0), Post-LN blocks (pre_lnorm False), per-layer head counts and FFN
widths, d_head = d_model / n_head, adaptive input embedding and adaptive softmax with tied
weights (tie_weight True), untied head projection and tied tail projections (tie_projs
[False, True, ...]), init N(0, 0.02) for weights and N(0, 0.01) for projections, dropout 0.1,
dropatt 0. Vocabulary: GPT-2 BPE (50257 tokens), cutoffs [19997, 39997] as in the archive after
truncation at the vocabulary size.

The attention and memory follow Dai et al. (2019); the block structure follows the NSC authors'
training script (github.com/Optima-CityU/neural-spectral-capacity, search/txl/train_wt103_txl.py,
MIT licence), adapted to adaptive embeddings, per-layer head counts and no extra final LayerNorm.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CUTOFFS = (19997, 39997)
N_TOKEN = 50257
INIT_STD, PROJ_INIT_STD = 0.02, 0.01


def clusters(n_token=N_TOKEN, cutoffs=CUTOFFS):
    """(start, end) token ranges of the head and tail clusters."""
    ends = list(cutoffs) + [n_token]
    return list(zip([0] + ends[:-1], ends))


class AdaptiveEmbedding(nn.Module):
    """Adaptive input embedding (Baevski and Auli, 2019) in the Transformer-XL form.

    div_val = 1: one table of width d_embed, projected to d_model unless d_embed == d_model.
    div_val > 1: cluster i has width d_embed // div_val**i and its own projection.
    Outputs are scaled by sqrt(d_model) as in Transformer-XL.
    """

    def __init__(self, d_model, d_embed, div_val):
        super().__init__()
        self.d_model, self.d_embed, self.div_val = d_model, d_embed, div_val
        self.ranges = clusters()
        if div_val == 1:
            self.tables = nn.ParameterList([nn.Parameter(torch.randn(N_TOKEN, d_embed) * INIT_STD)])
            self.projs = nn.ParameterList(
                [nn.Parameter(torch.randn(d_model, d_embed) * PROJ_INIT_STD)] if d_embed != d_model else [])
        else:
            widths = [d_embed // div_val ** i for i in range(len(self.ranges))]
            self.tables = nn.ParameterList([nn.Parameter(torch.randn(b - a, w) * INIT_STD)
                                            for (a, b), w in zip(self.ranges, widths)])
            self.projs = nn.ParameterList([nn.Parameter(torch.randn(d_model, w) * PROJ_INIT_STD) for w in widths])

    def forward(self, x):
        if self.div_val == 1:
            h = F.embedding(x, self.tables[0])
            if len(self.projs):
                h = F.linear(h, self.projs[0])
        else:
            h = x.new_zeros(*x.shape, self.d_model, dtype=self.projs[0].dtype)
            for (a, b), table, proj in zip(self.ranges, self.tables, self.projs):
                m = (x >= a) & (x < b)
                if m.any():
                    h[m] = F.linear(F.embedding(x[m] - a, table), proj).to(h.dtype)
        return h * math.sqrt(self.d_model)


class AdaptiveLogSoftmax(nn.Module):
    """Adaptive softmax (Grave et al., 2017) tied to an AdaptiveEmbedding.

    Output weights are the embedding tables. The head projection is untied (its own matrix); tail
    projections are tied to the input projections. Returns per-token negative log-likelihood.
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        n_tail = len(emb.ranges) - 1
        self.cluster_weight = nn.Parameter(torch.zeros(n_tail, emb.tables[0].shape[1]))
        self.cluster_bias = nn.Parameter(torch.zeros(n_tail))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(b - a)) for a, b in emb.ranges])
        need_head_proj = emb.div_val > 1 or emb.d_embed != emb.d_model
        self.head_proj = nn.Parameter(torch.randn(emb.d_model, emb.tables[0].shape[1]) * PROJ_INIT_STD) \
            if need_head_proj else None

    def _cluster(self, i):
        """Output weight, bias and projection (or None) of cluster i."""
        emb = self.emb
        a, b = emb.ranges[i]
        if emb.div_val == 1:
            return emb.tables[0][a:b], self.biases[i], self.head_proj
        return emb.tables[i], self.biases[i], self.head_proj if i == 0 else emb.projs[i]

    def forward(self, h, target):
        h, target = h.reshape(-1, h.shape[-1]), target.reshape(-1)
        w0, b0, p0 = self._cluster(0)
        z0 = F.linear(h, p0.t()) if p0 is not None else h
        head = F.linear(z0, torch.cat([w0, self.cluster_weight]), torch.cat([b0, self.cluster_bias]))
        head_lp = F.log_softmax(head.float(), dim=-1)
        n_head = w0.shape[0]
        nll = torch.zeros_like(target, dtype=torch.float)
        m = target < n_head
        nll[m] = -head_lp[m].gather(1, target[m, None]).squeeze(1)
        for i in range(1, len(self.emb.ranges)):
            a, b = self.emb.ranges[i]
            m = (target >= a) & (target < b)
            if not m.any():
                continue
            w, bias, p = self._cluster(i)
            z = F.linear(h[m], p.t()) if p is not None else h[m]
            lp = F.log_softmax(F.linear(z, w, bias).float(), dim=-1)
            nll[m] = -(head_lp[m, n_head + i - 1] + lp.gather(1, (target[m] - a)[:, None]).squeeze(1))
        return nll


class RelAttention(nn.Module):
    """Relative multi-head attention with memory (Transformer-XL, attn_type 0)."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        inner = n_head * self.d_head
        self.qkv = nn.Linear(d_model, 3 * inner, bias=False)
        self.r = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.u = nn.Parameter(torch.zeros(n_head, self.d_head))
        self.v = nn.Parameter(torch.zeros(n_head, self.d_head))
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def rel_shift(x):
        b, n, q, k = x.shape
        x = torch.cat([x.new_zeros(b, n, q, 1), x], dim=-1).view(b, n, k + 1, q)
        return x[:, :, 1:].view(b, n, q, k)

    def forward(self, h, mem, pos, mask):
        b, q, _ = h.shape
        cat = torch.cat([mem, h], 1) if mem is not None else h
        k = cat.shape[1]
        qkv = self.qkv(cat).view(b, k, 3, self.n_head, self.d_head)
        Q = qkv[:, -q:, 0].transpose(1, 2)
        K = qkv[:, :, 1].transpose(1, 2)
        V = qkv[:, :, 2].transpose(1, 2)
        R = self.r(pos).view(k, self.n_head, self.d_head).permute(1, 2, 0)
        ac = (Q + self.u[:, None]) @ K.transpose(-1, -2)
        bd = self.rel_shift((Q + self.v[:, None]) @ R)
        att = ((ac + bd) / math.sqrt(self.d_head)).masked_fill(mask, float('-inf')).softmax(-1)
        return self.drop(self.o((att @ V).transpose(1, 2).reshape(b, q, -1)))


class Block(nn.Module):
    """Post-LN Transformer-XL block: h = LN(h + Attn(h)); h = LN(h + FFN(h))."""

    def __init__(self, d_model, n_head, d_inner, dropout):
        super().__init__()
        self.attn = RelAttention(d_model, n_head, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(d_inner, d_model), nn.Dropout(dropout))

    def forward(self, h, mem, pos, mask):
        h = self.ln1(h + self.attn(h, mem, pos, mask))
        return self.ln2(h + self.ff(h))


class TransformerXL(nn.Module):
    """Decoder-only Transformer-XL language model described by an archive-style config dict."""

    def __init__(self, cfg, dropout=0.1):
        super().__init__()
        d = cfg['d_model']
        self.d_model, self.mem_len = d, cfg['mem_len']
        self.emb = AdaptiveEmbedding(d, cfg['d_embed'], cfg['div_val'])
        self.crit = AdaptiveLogSoftmax(self.emb)
        self.blocks = nn.ModuleList(Block(d, h, f, dropout) for h, f in zip(cfg['n_head'], cfg['d_inner']))
        self.drop = nn.Dropout(dropout)
        inv = 1.0 / (10000 ** (torch.arange(0, d, 2.0) / d))
        self.register_buffer('inv_freq', inv, persistent=False)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, mems):
        """Return per-token NLL and the new memories. mems: list (one per block) or None."""
        b, q = x.shape
        m = 0 if mems is None else mems[0].shape[1]
        k = q + m
        mask = torch.triu(torch.ones(q, k, dtype=torch.bool, device=x.device), diagonal=1 + m)
        t = torch.arange(k - 1, -1, -1.0, device=x.device)
        s = t[:, None] * self.inv_freq[None]
        pos = self.drop(torch.cat([s.sin(), s.cos()], -1).to(self.emb.tables[0].dtype))
        h = self.drop(self.emb(x))
        new = []
        for i, blk in enumerate(self.blocks):
            mem = None if mems is None else mems[i]
            new.append((torch.cat([mem, h], 1) if mem is not None else h)[:, -self.mem_len:].detach())
            h = blk(h, mem, pos, mask)
        return self.crit(self.drop(h), y), new

    def param_counts(self):
        """Total, embedding (tables, projections, softmax extras) and decoder parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        emb = sum(p.numel() for p in self.emb.parameters()) + sum(p.numel() for p in self.crit.parameters()
                                                                   if all(p is not q for q in self.emb.parameters()))
        return {'total': total, 'embedding': emb, 'decoder': total - emb}

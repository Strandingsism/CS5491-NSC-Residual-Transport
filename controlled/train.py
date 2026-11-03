"""Train one archive-style Transformer-XL on WikiText-103 (GPT-2 BPE) for a fixed token budget.

The configuration uses the archive's fields (d_model, n_head list, d_inner list, d_embed, div_val,
mem_len). Training: Adam, linear warmup then cosine decay, gradient clipping 0.25, bf16 autocast,
segments of tgt_len = mem_len = 192 tokens carried with memory across steps (Transformer-XL
streaming). Validation PPL is computed on the full validation split with memory.

Run from project root, for example:
  .venv/Scripts/python.exe controlled/train.py --name pilot --d_model 256 --n_layer 8 --n_head 8 \
      --d_inner 1024 --d_embed 256 --div_val 1 --tokens 5e6 --seed 0
Writes controlled/runs/<name>.json (config, parameter counts, losses, valid PPL, wall time).
"""
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import torch

from txl import TransformerXL

HERE = Path(__file__).resolve().parent


def streams(ids, bsz):
    """Cut a token array into bsz parallel streams (Transformer-XL batching)."""
    n = len(ids) // bsz
    return torch.from_numpy(ids[:n * bsz].astype(np.int64)).view(bsz, n)


@torch.no_grad()
def evaluate(model, data, tgt_len, max_tokens=None):
    """Mean NLL (nats/token) over the stream, with memory carried between segments."""
    model.eval()
    mems, total, count = None, 0.0, 0
    end = data.shape[1] - 1 if max_tokens is None else min(data.shape[1] - 1, max_tokens // data.shape[0])
    for i in range(0, end, tgt_len):
        x = data[:, i:min(i + tgt_len, end)].cuda()
        y = data[:, i + 1:min(i + tgt_len, end) + 1].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            nll, mems = model(x, y, mems)
        total += nll.sum().item()
        count += nll.numel()
    model.train()
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--d_model', type=int, required=True)
    p.add_argument('--n_layer', type=int, required=True)
    p.add_argument('--n_head', type=int, required=True, help='heads per layer (uniform)')
    p.add_argument('--d_inner', type=int, required=True, help='FFN width per layer (uniform)')
    p.add_argument('--d_embed', type=int, required=True)
    p.add_argument('--div_val', type=int, default=1)
    p.add_argument('--tokens', type=float, required=True, help='training tokens')
    p.add_argument('--batch', type=int, default=32, help='parallel streams')
    p.add_argument('--tgt_len', type=int, default=192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup', type=float, default=0.05, help='warmup fraction of steps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_tokens', type=float, default=None, help='cap validation tokens (pilot only)')
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = {'d_model': a.d_model, 'n_head': [a.n_head] * a.n_layer, 'd_inner': [a.d_inner] * a.n_layer,
           'd_embed': a.d_embed, 'div_val': a.div_val, 'mem_len': a.tgt_len}
    model = TransformerXL(cfg).cuda()
    counts = model.param_counts()
    train = streams(np.load(HERE / 'data' / 'train.npy', mmap_mode='r'), a.batch)
    valid = streams(np.load(HERE / 'data' / 'valid.npy'), 16)
    steps = int(a.tokens // (a.batch * a.tgt_len))
    warm = max(1, int(a.warmup * steps))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    print(f'{a.name}: params {counts}, {steps} steps of {a.batch}x{a.tgt_len} tokens')

    mems, log, pos = None, [], 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        if pos + a.tgt_len + 1 > train.shape[1]:
            pos, mems = 0, None
        x = train[:, pos:pos + a.tgt_len].cuda(non_blocking=True)
        y = train[:, pos + 1:pos + a.tgt_len + 1].cuda(non_blocking=True)
        pos += a.tgt_len
        with torch.autocast('cuda', dtype=torch.bfloat16):
            nll, mems = model(x, y, mems)
            loss = nll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()
        if (step + 1) % max(1, steps // 20) == 0 or step == steps - 1:
            el = time.perf_counter() - t0
            log.append({'step': step + 1, 'loss': loss.item(), 'seconds': el})
            print(f'  step {step + 1}/{steps}  loss {loss.item():.3f}  {(step + 1) * a.batch * a.tgt_len / el:,.0f} tok/s')
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    v = evaluate(model, valid, a.tgt_len, None if a.eval_tokens is None else int(a.eval_tokens))
    out = {'name': a.name, 'config': cfg, 'args': vars(a), 'params': counts, 'steps': steps,
           'train_seconds': train_s, 'eval_seconds': time.perf_counter() - t1, 'valid_nll': v,
           'valid_ppl': math.exp(v), 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9, 'log': log}
    print(f'  valid ppl {math.exp(v):.2f}  train {train_s:.0f}s  peak {out["peak_mem_gb"]:.1f} GB')
    (HERE / 'runs').mkdir(exist_ok=True)
    (HERE / 'runs' / f'{a.name}.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

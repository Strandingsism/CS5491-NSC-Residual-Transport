"""Run the pre-registered controlled panel (README section 3) one training run at a time.

Panel A: d_model 256, 8 layers, 8 heads, FFN 1024; d_embed {64,128,256,512} x div_val {1,2,4}; seeds 0-2.
Panel B: d_model 512, 8 layers, 8 heads, FFN 2048; d_embed {128,256,512}, div_val 1; seeds 0-2.
Panel C: training-length check, panel A d_embed {64,256}, div_val 1, seed 0, 6e7 tokens.
All other runs use 2e7 tokens. Seeds are interleaved so that a partial panel stays balanced.
Finished runs (runs/<name>.json present) are skipped, so the script can be restarted.

Run from project root:  python controlled/run_panel.py [--panel A B C] [--dry]
"""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TOKENS, LONG_TOKENS = 2e7, 6e7
DECODERS = {'A': dict(d_model=256, n_layer=8, n_head=8, d_inner=1024),
            'B': dict(d_model=512, n_layer=8, n_head=8, d_inner=2048)}


def runs():
    """Ordered list of (name, train.py arguments)."""
    out = []
    for seed in range(3):
        for de in (64, 128, 256, 512):
            for dv in (1, 2, 4):
                out.append((f'A_e{de}_v{dv}_s{seed}', dict(DECODERS['A'], d_embed=de, div_val=dv, seed=seed, tokens=TOKENS)))
        for de in (128, 256, 512):
            out.append((f'B_e{de}_v1_s{seed}', dict(DECODERS['B'], d_embed=de, div_val=1, seed=seed, tokens=TOKENS)))
    for de in (64, 256):
        out.append((f'C_e{de}_v1_s0', dict(DECODERS['A'], d_embed=de, div_val=1, seed=0, tokens=LONG_TOKENS)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', nargs='+', default=['A', 'B', 'C'])
    p.add_argument('--dry', action='store_true', help='list pending runs only')
    a = p.parse_args()
    todo = [(n, kw) for n, kw in runs() if n[0] in a.panel and not (HERE / 'runs' / f'{n}.json').exists()]
    print(f'{len(todo)} runs pending')
    for i, (name, kw) in enumerate(todo, 1):
        cmd = [sys.executable, str(HERE / 'train.py'), '--name', name] + [x for k, v in kw.items() for x in (f'--{k}', str(v))]
        print(f'[{i}/{len(todo)}] {" ".join(cmd[2:])}', flush=True)
        if not a.dry:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

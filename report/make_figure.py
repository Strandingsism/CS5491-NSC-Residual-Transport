"""Figure 1 of the report: predicted vs Monte Carlo transport factors T_s.

(a) archive setting (sigma=0.02, n=mem=192), deepest d=256 configuration of the strengthened
    validation (10 seeds, error bars = 2 standard errors);
(b) strong-signal check (sigma=0.04, d=128, f=512, L=8; not an archive setting).
Reads only label-free outputs. Run from project root:
  .venv/Scripts/python.exe report/make_figure.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MF = ROOT / 'analysis' / 'meanfield_transport'
COLORS = {'post': '#1f5fa8', 'pre': '#c0392b', 'none': '#7f8c8d'}
LABELS = {'post': 'Post-LN (archive)', 'pre': 'Pre-LN', 'none': 'no norm'}

rows = json.loads((MF / 'strengthen_rows.json').read_text())
key = max((r for r in rows if r['d'] == 256), key=lambda r: r['L'])['key']
strong = json.loads((MF / 'mechanism_checks.json').read_text())['strong_regime'][:3]

plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
ax = axes[0]
for r in (r for r in rows if r['key'] == key):
    M = np.array(r['measured_seeds'])
    x = np.arange(1, M.shape[1] + 1)
    se = M.std(0, ddof=1) / np.sqrt(len(M))
    ax.errorbar(x, M.mean(0), yerr=2 * se, fmt='o', ms=2.2, lw=0.6, color=COLORS[r['norm']], label=LABELS[r['norm']])
    ax.plot(x, r['v2'], '-', lw=0.9, color=COLORS[r['norm']])
ax.set_title(r'(a) archive setting: d=256, L=16, $\sigma$=0.02', fontsize=8)
ax.set_xlabel('sublayer s (attn$_1$, ffn$_1$, ..., ffn$_L$)')
ax.set_ylabel('transport factor $T_s$')
ax.legend(frameon=False, fontsize=7)
ax = axes[1]
for r in strong:
    x = np.arange(1, len(r['measured']) + 1)
    ax.plot(x, r['measured'], 'o', ms=2.2, color=COLORS[r['norm']])
    ax.plot(x, r['predicted'], '-', lw=0.9, color=COLORS[r['norm']], label=LABELS[r['norm']])
ax.set_title(r'(b) strong-signal check: d=128, L=8, $\sigma$=0.04', fontsize=8)
ax.set_xlabel('sublayer s')
for a in axes:
    a.spines[['top', 'right']].set_visible(False)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name('transport_profiles.pdf'))
if len(__import__('sys').argv) > 1:
    fig.savefig(__import__('sys').argv[1], dpi=160)
print(key)

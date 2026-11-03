"""Figure 1 of the report.

(a) Study 1 (v2): predicted vs Monte Carlo transport factors T_s in the archive setting (sigma=0.02,
    n=mem=192), deepest d=256 development configuration of the strengthened validation (10 seeds,
    error bars = 2 standard errors). Label-free.
(b) Study 2 (v3): development residual of log PPL after a within-width fit on log NSC, averaged by
    d_embed, for each d_model. Reads development labels only.

Run from project root:
  .venv/Scripts/python.exe report/make_figure.py            # writes report/figure1.pdf
  .venv/Scripts/python.exe report/make_figure.py out.png    # also writes a PNG preview
"""
from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nsc-starter'))
from nsc import original_nsc  # noqa: E402

MF = ROOT / 'analysis' / 'meanfield_transport'
COLORS = {'post': '#1f5fa8', 'pre': '#c0392b', 'none': '#7f8c8d'}
LABELS = {'post': 'Post-LN (archive)', 'pre': 'Pre-LN', 'none': 'no norm'}
WIDTH_COLORS = {64: '#9ecae1', 128: '#4292c6', 256: '#2171b5', 512: '#08306b'}

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))

# (a) transport factors
rows = json.loads((MF / 'strengthen_rows.json').read_text())
key = max((r for r in rows if r['d'] == 256), key=lambda r: r['L'])['key']
ax = axes[0]
for r in (r for r in rows if r['key'] == key):
    M = np.array(r['measured_seeds'])
    x = np.arange(1, M.shape[1] + 1)
    se = M.std(0, ddof=1) / np.sqrt(len(M))
    ax.errorbar(x, M.mean(0), yerr=2 * se, fmt='o', ms=2, lw=0.6, color=COLORS[r['norm']], label=LABELS[r['norm']])
    ax.plot(x, r['v2'], '-', lw=0.9, color=COLORS[r['norm']])
ax.set_title(r'(a) Study 1: transport $T_s$ ($d$=256, $L$=16)', fontsize=9)
ax.set_xlabel('sublayer $s$')
ax.set_ylabel('$T_s$ (lines: prediction)')
ax.legend(frameon=False, fontsize=7.5)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# (b) embedding-width residuals (development labels)
D = ROOT / 'nsc-starter' / 'data'
configs = json.loads((D / 'architectures.json').read_text())
keys = json.loads((D / 'splits.json').read_text())['development']
lab = json.loads((D / 'development_labels.json').read_text())
ax = axes[1]
for w in (64, 128, 256, 512):
    ks = [k for k in keys if configs[k]['d_model'] == w]
    y = np.array([math.log(lab[k]['valid_ppl']) for k in ks])
    X = np.column_stack([np.ones(len(ks)), [math.log(original_nsc(configs[k])) for k in ks]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    de = np.array([configs[k]['d_embed'] for k in ks])
    xs = [v for v in (128, 256, 512) if (de == v).any()]
    ax.plot(xs, [res[de == v].mean() for v in xs], 'o-', ms=3, lw=1.1, color=WIDTH_COLORS[w], label=f'$d$={w}')
    if w in (128, 256):
        ax.axvline(w, color=WIDTH_COLORS[w], lw=0.6, ls=':')
ax.axhline(0, color='0.6', lw=0.5)
ax.set_xscale('log', base=2)
ax.set_xticks([128, 256, 512])
ax.set_xticklabels(['128', '256', '512'])
ax.set_title('(b) Study 2: residual vs. embedding width', fontsize=9)
ax.set_xlabel('$d_{\\mathrm{embed}}$ (dotted: $d_{\\mathrm{embed}}=d$)')
ax.set_ylabel('mean residual of log PPL')
ax.legend(frameon=False, fontsize=7.5, ncol=2)
for a in axes:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name('figure1.pdf'))
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=160)

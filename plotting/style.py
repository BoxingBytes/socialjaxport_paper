import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = 'out'
DEFECTOR_COLOR = '#D55E00'
COOPERATOR_COLOR = '#0072B2'
AVERAGE_COLOR = '#555555'
BOOTSTRAP_RESAMPLES = 10000
CI_LEVEL = 0.95

RC = {
    'figure.figsize': (4.2, 3.2),
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#DDDDDD',
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}


def use_style():
    plt.rcParams.update(RC)


def bootstrap_ci(samples, rng, resamples=BOOTSTRAP_RESAMPLES, level=CI_LEVEL):
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 2:
        value = samples.mean() if len(samples) else np.nan
        return value, value, value
    draws = rng.integers(len(samples), size=(resamples, len(samples)))
    means = samples[draws].mean(axis=1)
    tail = 100*(1 - level)/2
    return samples.mean(), np.percentile(means, tail), np.percentile(means, 100 - tail)


def plot_with_ci(ax, x, mean, low, high, color, label, marker='o', **kwargs):
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    return ax.plot(x, mean, color=color, label=label, marker=marker,
                   markeredgecolor='white', markeredgewidth=0.6, **kwargs)[0]


def save(fig, name, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return path

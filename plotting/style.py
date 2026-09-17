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


CURVE_PALETTE = ('#CE93D8', '#4CAF50', '#F0932B', '#22B8CF', '#5B9BD5', '#E8736C',
                 '#8D6E63', '#607D8B', '#9CCC65')
METHOD_LABELS = {
    'ippo_IR': 'IPPO: Individual Reward',
    'ippo_SR': 'IPPO: Common Reward',
    'ippo_CR': 'IPPO: Common Reward',
    'ippo_RE': 'IPPO-RE',
    'mappo': 'MAPPO',
    'vdn': 'VDN',
    'svo': 'SVO',
}
ENV_LABELS = {
    'coins': 'Coins',
    'common_harvest': 'Commons Harvest: Open',
    'common_harvest_closed': 'Commons Harvest: Closed',
    'common_harvest_partnership': 'Commons Harvest: Partnership',
    'clean_up': 'Clean Up',
    'coop_mining': 'Coop Mining',
    'mushrooms': 'Mushrooms',
    'gift_refinement': 'Gift Refinement',
    'prisoners_dilemma_arena': 'Prisoners Dilemma: Arena',
}
METHOD_ORDER = ('ippo_IR', 'ippo_SR', 'ippo_CR', 'mappo', 'vdn', 'svo', 'ippo_RE')
PANEL_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def method_label(method):
    return METHOD_LABELS.get(method, method.replace('_', ' '))


def env_label(env):
    return ENV_LABELS.get(env, env.replace('_', ' ').title())


# Fixed slot per known method so colors stay identical across figures and filters.
def method_colors(methods):
    free = [c for i, c in enumerate(CURVE_PALETTE)
            if i >= len(METHOD_ORDER) or METHOD_ORDER[i] not in methods]
    colors = {}
    for method in sort_methods(methods):
        if method in METHOD_ORDER:
            colors[method] = CURVE_PALETTE[METHOD_ORDER.index(method) % len(CURVE_PALETTE)]
        else:
            colors[method] = free.pop(0) if free else CURVE_PALETTE[len(colors) % len(CURVE_PALETTE)]
    return colors


def sort_methods(methods):
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(set(methods) - set(known))


def panel_grid(num_panels, max_cols=3, panel_size=(3.4, 2.4)):
    ncols = min(max_cols, num_panels)
    nrows = -(-num_panels//ncols)
    figsize = (panel_size[0]*ncols, panel_size[1]*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout='constrained')
    flat = axes.ravel().tolist()
    for ax in flat[num_panels:]:
        ax.set_visible(False)
    return fig, flat[:num_panels]


def scientific_xaxis(ax):
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=False)
    ax.xaxis.get_offset_text().set_size(plt.rcParams['xtick.labelsize'])


def panel_caption(ax, index, text, single=False):
    caption = text if single else f'({PANEL_LETTERS[index]}) {text}'
    ax.set_title(caption, y=-0.42, fontsize=plt.rcParams['axes.titlesize'] + 1)


def bottom_legend(fig, handles, labels, max_cols=6):
    fig.legend(handles, labels, loc='outside lower center', frameon=False,
               ncol=min(max_cols, len(handles)))

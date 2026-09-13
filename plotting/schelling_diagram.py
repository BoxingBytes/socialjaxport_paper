import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(REPO_ROOT, 'out', 'schelling.json')
FIGURE_NAME = 'schelling_diagram.png'
CI_SEED = 0


def role_means(episode, num_agents):
    returns = np.array(episode['per_agent_returns'])
    cooperators = np.zeros(num_agents, dtype=bool)
    cooperators[episode['cooperator_indices']] = True
    return (returns[cooperators].mean() if cooperators.any() else None,
            returns[~cooperators].mean() if (~cooperators).any() else None,
            returns.mean())


def observations(payload):
    num_agents = payload['config']['num_agents']
    cooperator, defector, average = {}, {}, {}
    for episode in payload['episodes']:
        j = episode['j']
        c, d, a = role_means(episode, num_agents)
        if c is not None:
            cooperator.setdefault(j - 1, {})[episode['seed']] = c
        if d is not None:
            defector.setdefault(j, {})[episode['seed']] = d
        average.setdefault(j*(num_agents - 1)/num_agents, {})[episode['seed']] = a
    return num_agents, cooperator, defector, average


def curve(series, rng):
    xs = sorted(series)
    stats = np.array([style.bootstrap_ci(list(series[x].values()), rng) for x in xs])
    return np.array(xs), stats[:, 0], stats[:, 1], stats[:, 2]


# Paired by seed: common random numbers make the same seed the same apple stream across compositions.
def gap(cooperator, defector, x, rng):
    seeds = sorted(set(cooperator[x]) & set(defector[x]))
    mean, low, high = style.bootstrap_ci([defector[x][s] - cooperator[x][s] for s in seeds], rng)
    return mean, low, high, low > 0


def mean_of(series, x):
    return float(np.mean(list(series[x].values())))


# Hughes et al. (2018) sequential social dilemma conditions, R_c(N) read at x = N - 1.
def conditions(num_agents, cooperator, defector, rng):
    mutual_cooperation = mean_of(cooperator, num_agents - 1)
    fear = gap(cooperator, defector, 0, rng)
    greed = gap(cooperator, defector, num_agents - 1, rng)
    fmt = lambda g: f'{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]'
    return {'Rc(N) > Rd(0)  mutual cooperation beats mutual defection':
                mutual_cooperation > mean_of(defector, 0),
            'Rc(N) > Rc(0)  mutual cooperation beats being exploited':
                mutual_cooperation > mean_of(cooperator, 0),
            f'fear or greed  (fear D-C {fmt(fear)}, greed D-C {fmt(greed)})':
                fear[3] or greed[3]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=JSON_PATH)
    parser.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'out'))
    args = parser.parse_args()

    with open(args.data) as f:
        payload = json.load(f)
    num_agents, cooperator, defector, average = observations(payload)

    rng = np.random.default_rng(CI_SEED)
    style.use_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for series, color, label, kwargs in (
            (defector, style.DEFECTOR_COLOR, 'defector', {}),
            (cooperator, style.COOPERATOR_COLOR, 'cooperator', {}),
            (average, style.AVERAGE_COLOR, 'average', {'linestyle': '--', 'marker': None})):
        style.plot_with_ci(ax, *curve(series, rng), color=color, label=label, **kwargs)

    ax.set_xlabel('number of other cooperators')
    ax.set_ylabel('individual payoff (apples / episode)')
    ax.set_xticks(range(num_agents))
    ax.set_xlim(-0.15, num_agents - 1 + 0.15)
    ax.legend(frameon=False, loc='upper left')
    path = style.save(fig, FIGURE_NAME, args.out_dir)

    for name, holds in conditions(num_agents, cooperator, defector, rng).items():
        print(f'{"PASS" if holds else "FAIL"}  {name}')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()

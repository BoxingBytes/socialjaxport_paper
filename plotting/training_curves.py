import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curves
import style

METRIC = 'env/episode_return'
CI_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', help='comma separated env names (default: all found in out/curves)')
    parser.add_argument('--methods', help='comma separated method names (default: all found)')
    parser.add_argument('--seeds', help='comma separated seeds (default: all found)')
    parser.add_argument('--metric', default=METRIC)
    parser.add_argument('--x', default='agent_steps', choices=curves.X_AXES,
                        help='row key for the x axis')
    parser.add_argument('--truncate', action='store_true',
                        help='cut every method of a panel at the shortest one (default: keep '
                             'each method its full length)')
    parser.add_argument('--ylabel', default='Returns')
    parser.add_argument('--xlabel')
    parser.add_argument('--collective', action='store_true',
                        help='scale the per-agent metric by env.num_agents')
    parser.add_argument('--curves-dir', default=curves.CURVES_DIR)
    parser.add_argument('--out', default='training_curves.png')
    return parser.parse_args()


def split(value, cast=str):
    return None if value is None else [cast(v) for v in value.split(',') if v]


def agent_scale(runs):
    counts = {run.header['args']['env']['num_agents'] for run in runs}
    if len(counts) > 1:
        print(f'WARNING: mixed num_agents {sorted(counts)}; collective scaling uses the maximum')
    return float(max(counts))


# Seeds of one method always share a stop; methods only do under --truncate.
def method_stops(by_method, x_key, truncate):
    ends = {method: min(curves.last_x(run, x_key) for run in runs)
            for method, runs in by_method.items()}
    shortest = min(ends.values())
    ragged = {m: e for m, e in ends.items() if e > shortest*1.001}
    if truncate:
        return {m: shortest for m in ends}, shortest, ragged
    return ends, max(ends.values()), ragged


def panel(ax, env, by_method, args, colors, rng):
    stops, limit, ragged = method_stops(by_method, args.x, args.truncate)
    x_label, x_scale = curves.x_axis(args.x, limit)
    if ragged:
        longest = ', '.join(f'{m}={e*x_scale:.3g}' for m, e in sorted(ragged.items()))
        if args.truncate:
            print(f'WARNING: [{env}] truncating at {min(stops.values())*x_scale:.3g}; '
                  f'longer: {longest}')
        else:
            print(f'WARNING: [{env}] methods end at different x; longer: {longest} '
                  f'(pass --truncate to align them)')
    handles = {}
    for method, runs in by_method.items():
        scale = agent_scale(runs) if args.collective else 1.0
        aggregate = curves.aggregate(runs, args.metric, rng, x_key=args.x, x_stop=stops[method],
                                     scale=scale)
        if len(runs) < 2:
            print(f'WARNING: [{env}] {method} has a single seed, no confidence band drawn')
        handles[method] = style.plot_with_ci(ax, aggregate.x*x_scale, aggregate.mean,
                                             aggregate.low, aggregate.high, colors[method],
                                             style.method_label(method), marker=None)
    ax.set_xlabel(args.xlabel or x_label)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(0, limit*x_scale)
    if limit*x_scale >= 1e4:
        style.scientific_xaxis(ax)
    return handles


def main():
    args = parse_args()
    runs = curves.discover(args.curves_dir, split(args.envs), split(args.methods),
                           split(args.seeds, int))
    for run in runs:
        curves.load_rows(run)
    grouped = curves.group(runs)
    methods = style.sort_methods({run.method for run in runs})
    colors = style.method_colors(methods)
    rng = np.random.default_rng(CI_SEED)

    style.use_style()
    fig, axes = style.panel_grid(len(grouped))
    single = len(grouped) == 1
    handles = {}
    for index, ((env, by_method), ax) in enumerate(zip(grouped.items(), axes)):
        handles.update(panel(ax, env, by_method, args, colors, rng))
        if not single:
            style.panel_caption(ax, index, style.env_label(env))
    ordered = [m for m in methods if m in handles]
    style.bottom_legend(fig, [handles[m] for m in ordered],
                        [style.method_label(m) for m in ordered])
    path = style.save(fig, args.out)
    print(f'{len(runs)} runs, {len(grouped)} envs, {len(methods)} methods -> {path}')


if __name__ == '__main__':
    try:
        main()
    except curves.CurveDataError as error:
        sys.exit(f'ERROR: {error}')

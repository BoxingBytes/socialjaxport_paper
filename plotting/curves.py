"""Scraping and aggregation of out/curves/*.jsonl, shared by every curve figure."""
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_DIR = os.path.join(REPO_ROOT, 'out', 'curves')
SUFFIX = '.jsonl'
X_AXES = ('agent_steps', 'uptime')
TIME_UNITS = (('seconds', 1.0, 600), ('minutes', 1/60, 7200), ('hours', 1/3600, None))


def x_axis(x_key, largest):
    if x_key != 'uptime':
        return 'Environment timesteps', 1.0
    unit, scale, _ = next(u for u in TIME_UNITS if u[2] is None or largest < u[2])
    return f'Wall-clock time ({unit})', scale
SEED_RE = re.compile(r'^(?P<stem>.+)_s(?P<seed>\d+)$')


class CurveDataError(Exception):
    pass


class Run:
    def __init__(self, path, env, method, seed):
        self.path, self.env, self.method, self.seed = path, env, method, seed
        self.header, self.rows, self.final_eval = None, [], None

    def __repr__(self):
        return f'Run({self.env}/{self.method}/s{self.seed})'


def _fail(path, reason):
    raise CurveDataError(
        f'cannot parse run file {os.path.basename(path)!r} in {os.path.dirname(path)}\n'
        f'  reason: {reason}\n'
        f'  expected layout: <env_name>_<method>_s<seed>{SUFFIX}, '
        f'e.g. common_harvest_ippo_IR_s1{SUFFIX}\n'
        f'  the env/method split is taken from the file\'s "header" record '
        f'("env_name" and "condition"), and the filename must agree with it.')


def read_header(path):
    with open(path) as f:
        first = f.readline()
    if not first.strip():
        _fail(path, 'file is empty')
    try:
        record = json.loads(first)
    except json.JSONDecodeError as exc:
        _fail(path, f'first line is not valid JSON ({exc})')
    if record.get('kind') != 'header':
        _fail(path, f'first record has kind={record.get("kind")!r}, expected "header"')
    for key in ('env_name', 'condition', 'seed'):
        if key not in record:
            _fail(path, f'header record has no {key!r} field')
    return record


def parse_run(path):
    name = os.path.basename(path)
    if not name.endswith(SUFFIX):
        _fail(path, f'filename does not end with {SUFFIX}')
    stem = name[:-len(SUFFIX)]
    match = SEED_RE.match(stem)
    if match is None:
        _fail(path, f'{stem!r} does not end with a _s<seed> part, so no seed could be read')
    header = read_header(path)
    env, method, seed = header['env_name'], header['condition'], int(header['seed'])
    expected = f'{env}_{method}_s{seed}'
    if stem != expected:
        _fail(path, f'filename stem {stem!r} disagrees with the header, which describes '
                    f'env_name={env!r}, condition={method!r}, seed={header["seed"]!r} '
                    f'and so expects the stem {expected!r}')
    if int(match.group('seed')) != seed:
        _fail(path, f'seed {match.group("seed")!r} in the filename is not the header seed {seed!r}')
    run = Run(path, env, method, seed)
    run.header = header
    return run


def load_rows(run):
    with open(run.path) as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(run.path, f'line {number} is not valid JSON ({exc})')
            kind = record.pop('kind', None)
            if kind == 'row':
                run.rows.append(record)
            elif kind == 'final_eval':
                run.final_eval = record
            elif kind == 'failed':
                raise CurveDataError(
                    f'{os.path.basename(run.path)} records a failed run:\n'
                    f'  {record.get("error", "")}')
    if not run.rows:
        raise CurveDataError(f'{os.path.basename(run.path)} contains no "row" records')
    return run


def discover(curves_dir=CURVES_DIR, envs=None, methods=None, seeds=None):
    if not os.path.isdir(curves_dir):
        raise CurveDataError(f'no curves directory at {curves_dir}')
    names = sorted(n for n in os.listdir(curves_dir) if n.endswith(SUFFIX))
    if not names:
        raise CurveDataError(f'no {SUFFIX} files in {curves_dir}')
    runs = [parse_run(os.path.join(curves_dir, n)) for n in names]
    if envs is not None:
        runs = [r for r in runs if r.env in envs]
    if methods is not None:
        runs = [r for r in runs if r.method in methods]
    if seeds is not None:
        runs = [r for r in runs if r.seed in seeds]
    if not runs:
        raise CurveDataError(
            f'no run in {curves_dir} matches envs={envs}, methods={methods}, seeds={seeds}')
    return runs


def group(runs):
    grouped = OrderedDict()
    for run in sorted(runs, key=lambda r: (r.env, r.method, r.seed)):
        grouped.setdefault(run.env, OrderedDict()).setdefault(run.method, []).append(run)
    return grouped


def series(run, metric, x_key='agent_steps'):
    missing = [k for k in (metric, x_key) if k not in run.rows[0]]
    if missing:
        raise CurveDataError(
            f'{os.path.basename(run.path)} rows have no {missing} key(s); available keys: '
            f'{sorted(run.rows[0])}')
    x = np.array([row[x_key] for row in run.rows], dtype=float)
    y = np.array([row[metric] for row in run.rows], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


class Aggregate:
    def __init__(self, x, mean, low, high, seeds):
        self.x, self.mean, self.low, self.high, self.seeds = x, mean, low, high, seeds


# Rows are logged at the first tick past each grid boundary, so seeds jitter off the grid.
def aggregate(runs, metric, rng, x_key='agent_steps', x_stop=None, scale=1.0, ci=True):
    curves = [series(run, metric, x_key) for run in runs]
    reference = max(curves, key=lambda c: len(c[0]))[0]
    stop = min(c[0][-1] for c in curves) if x_stop is None else x_stop
    grid = reference[reference <= stop]
    if grid.size == 0 or grid[-1] < stop:
        grid = np.append(grid, stop)
    stacked = np.array([np.interp(grid, x, y) for x, y in curves])*scale
    if ci and len(runs) > 1:
        stats = np.array([style.bootstrap_ci(column, rng) for column in stacked.T])
        return Aggregate(grid, stats[:, 0], stats[:, 1], stats[:, 2], [r.seed for r in runs])
    mean = stacked.mean(axis=0)
    return Aggregate(grid, mean, mean, mean, [r.seed for r in runs])


def last_x(run, x_key='agent_steps'):
    return float(run.rows[-1][x_key])

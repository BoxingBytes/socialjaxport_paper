"""Run (condition x seed) PufferLib 5.0 trainings back to back, one JSONL curve per run.

Run from anywhere; ./puffer runs inside PUFFERLIB_DIR, outputs land in this repo's out/.
Assumes ./build.sh <ENV_NAME> has already produced ./puffer in PUFFERLIB_DIR.
"""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ENV_NAME = 'common_harvest'
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PUFFERLIB_DIR = os.path.join(REPO_ROOT, '..', 'pufferlib')
PUFFER_BIN = './puffer'

CONDITIONS = {
    'ippo_SR_bestHP1': {'train.learning_rate': 0.00511051994, 'train.ent_coef': 0.00496632233,
        'train.clip_coef': 1, 'train.vf_clip_coef': 0.196644962, 'train.max_grad_norm': 0.230315849},
    'ippo_SR_bestHP2': {'train.learning_rate': 0.00499677658, 'train.ent_coef': 0.000238814173,
        'train.clip_coef': 1, 'train.vf_clip_coef': 0.36401403, 'train.max_grad_norm': 0.176118419},
    'ippo_SR_bestHP3': {'train.learning_rate': 0.00621463032, 'train.ent_coef': 0.000200891183,
        'train.clip_coef': 1, 'train.vf_clip_coef': 0.110606357, 'train.max_grad_norm': 0.223187044},
    'ippo_SR_bestHP4': {'train.learning_rate': 0.00792048406, 'train.ent_coef': 0.000245985604,
        'train.clip_coef': 1, 'train.vf_clip_coef': 0.15916875, 'train.max_grad_norm': 0.177031964},
    'ippo_SR_bestHP5': {'train.learning_rate': 0.00500054145, 'train.ent_coef': 0.00667259563,
        'train.clip_coef': 1, 'train.vf_clip_coef': 0.215366215, 'train.max_grad_norm': 0.446299344},
}

SEEDS = (1, 2, 3)
TOTAL_TIMESTEPS = None
POINTS = 64
FINAL_EVAL_EPISODES = 10_000
ENV_RNG_STRIDE = 65_536  # > num_envs: disjoint env streams across seeds
OUT_DIR = os.path.join(REPO_ROOT, 'out', 'curves')
CHECKPOINT_DIR = os.path.join(REPO_ROOT, 'out', 'checkpoints')
PUFFER_LOG_DIR = os.path.join(REPO_ROOT, 'out', 'puffer_logs')


def run_id(condition, seed):
    return f'{ENV_NAME}_{condition}_s{seed}'


def run_path(condition, seed):
    return os.path.join(OUT_DIR, f'{run_id(condition, seed)}.jsonl')


def puffer_log_path(condition, seed):
    return os.path.join(PUFFER_LOG_DIR, ENV_NAME, f'{run_id(condition, seed)}.ini')


def puffer_checkpoint_dir(condition, seed):
    return os.path.join(CHECKPOINT_DIR, ENV_NAME, run_id(condition, seed))


def final_checkpoint_dir(condition, seed):
    return os.path.join(CHECKPOINT_DIR, run_id(condition, seed))


def is_complete(path):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(line.startswith('{"kind": "done"') for line in f)


def script_sha256():
    with open(os.path.realpath(__file__), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def pufferlib_commit():
    try:
        out = subprocess.run(['git', '-C', PUFFERLIB_DIR, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def puffer_command(condition, seed):
    overrides = {
        **CONDITIONS[condition],
        'base.run_id': run_id(condition, seed),
        'base.seed': seed,
        'env.rng': seed * ENV_RNG_STRIDE,
        'base.checkpoint_dir': CHECKPOINT_DIR,
        'base.checkpoint_interval': 0,
        'base.log_dir': PUFFER_LOG_DIR,
        'base.eval_episodes': FINAL_EVAL_EPISODES,
        'sweep.downsample': POINTS,
    }
    if TOTAL_TIMESTEPS is not None:
        overrides['train.total_timesteps'] = TOTAL_TIMESTEPS
    return [PUFFER_BIN, 'train'] + [f'--{key}={value}' for key, value in overrides.items()]


def parse_scalar(raw):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def read_puffer_log(path):
    sections = {}
    node = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                node = sections.setdefault(line[1:-1], {})
                continue
            key, _, value = line.partition('=')
            node[key.strip()] = value.strip()
    metrics = {key: [float(v) for v in raw.split(',')]
        for key, raw in sections.pop('metrics').items()}
    args = {name: {key: parse_scalar(v) for key, v in section.items()}
        for name, section in sections.items()}
    return args, metrics


def move_final_checkpoint(condition, seed):
    src_dir = puffer_checkpoint_dir(condition, seed)
    final = max(os.listdir(src_dir))
    dst_dir = final_checkpoint_dir(condition, seed)
    os.makedirs(dst_dir, exist_ok=True)
    os.replace(os.path.join(src_dir, final), os.path.join(dst_dir, final))
    os.removedirs(src_dir)


class Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w')

    def write(self, kind, payload):
        self._f.write(json.dumps({'kind': kind, **payload}, default=str) + '\n')
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()


def run(condition, seed):
    for stale in (puffer_checkpoint_dir(condition, seed), final_checkpoint_dir(condition, seed)):
        shutil.rmtree(stale, ignore_errors=True)
    log_path = puffer_log_path(condition, seed)
    if os.path.exists(log_path):
        os.remove(log_path)

    command = puffer_command(condition, seed)
    header = {
        'env_name': ENV_NAME,
        'condition': condition,
        'seed': seed,
        'hostname': socket.gethostname(),
        'started_at': now_iso(),
        'pufferlib_commit': pufferlib_commit(),
        'script_sha256': script_sha256(),
        'command': command,
        'points': POINTS,
    }

    start = time.time()
    returncode = subprocess.run(command, cwd=PUFFERLIB_DIR).returncode
    wall_time = time.time() - start

    writer = Writer(run_path(condition, seed))
    if returncode != 0 or not os.path.exists(log_path):
        writer.write('header', header)
        writer.write('failed', {'error': f'{PUFFER_BIN} exited with code {returncode}',
            'puffer_log_exists': os.path.exists(log_path)})
        writer.close()
        return False

    args, metrics = read_puffer_log(log_path)
    assert args['base']['env_name'] == ENV_NAME, \
        f'{PUFFER_BIN} is built for {args["base"]["env_name"]!r}, rebuild with ./build.sh {ENV_NAME}'
    writer.write('header', {**header, 'total_timesteps': args['train']['total_timesteps'],
        'args': args})

    # Last column is the final snapshot with the eval overlaid
    columns = [dict(zip(metrics, values)) for values in zip(*metrics.values())]
    for row in columns[:-1]:
        writer.write('row', row)
    writer.write('final_eval', columns[-1])

    move_final_checkpoint(condition, seed)
    writer.write('done', {
        'wall_time_s': wall_time,
        'final_step': columns[-1]['agent_steps'],
        'finished_at': now_iso(),
    })
    writer.close()
    return True


def driver(argv):
    global PUFFERLIB_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
        help='Skip runs whose JSONL already has a done record')
    parser.add_argument('--only', type=str, default=None,
        help='Comma-separated subset of conditions')
    parser.add_argument('--seeds', type=str, default=None,
        help='Comma-separated subset of seeds')
    parser.add_argument('--pufferlib', type=str, default=PUFFERLIB_DIR,
        help='PufferLib root holding the built ./puffer')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    PUFFERLIB_DIR = os.path.abspath(args.pufferlib)
    puffer_path = os.path.normpath(os.path.join(PUFFERLIB_DIR, PUFFER_BIN))

    conditions = args.only.split(',') if args.only else list(CONDITIONS)
    for c in conditions:
        assert c in CONDITIONS, f'unknown condition {c!r}'
        assert not any(k.startswith('base.env_name') for k in CONDITIONS[c]), \
            'one build serves one env; drop the env_name override'
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(SEEDS)

    plan = list(itertools.product(conditions, seeds))
    skipped = {(c, s) for c, s in plan if args.resume and is_complete(run_path(c, s))}

    if args.dry_run:
        for condition, seed in plan:
            mark = 'skip' if (condition, seed) in skipped else 'write'
            print(f'{mark:>5}  {run_path(condition, seed)}')
            print(f'       {" ".join(puffer_command(condition, seed))}')
        found = os.path.exists(puffer_path)
        print(f'binary: {puffer_path} {"ok" if found else f"missing, build with ./build.sh {ENV_NAME}"}')
        return 0

    assert os.path.exists(puffer_path), f'no {puffer_path}: build with ./build.sh {ENV_NAME}'

    results = []
    durations = []
    for i, (condition, seed) in enumerate(plan):
        if (condition, seed) in skipped:
            results.append((condition, seed, 'skipped', 0.0))
            continue

        remaining = len(plan) - len(skipped) - len(durations)
        eta = f'{sum(durations) / len(durations) * remaining / 60:.0f} min' if durations else '?'
        print(f'\n=== [{i + 1}/{len(plan)}] {condition} seed {seed} (ETA {eta}) ===', flush=True)

        start = time.time()
        ok = run(condition, seed)
        elapsed = time.time() - start
        durations.append(elapsed)
        results.append((condition, seed, 'ok' if ok else 'failed', elapsed))

    print('\n=== summary ===')
    for condition, seed, status, elapsed in results:
        print(f'{status:>8}  {condition:<12} s{seed}  {elapsed / 60:6.1f} min')

    return 1 if any(status == 'failed' for _, _, status, _ in results) else 0


if __name__ == '__main__':
    sys.exit(driver(sys.argv[1:]))

"""Run (condition x seed) PufferLib trainings back to back, one JSONL curve per run.

Standalone: scp next to PufferLib on the GPU box and run from the PufferLib root.
Assumes ./build.sh <ENV_NAME> has already produced a CUDA build.
"""
import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ENV_NAME = 'common_harvest'

CONDITIONS = {
    'ippo_IR': {'env.shared_rewards': False},
    'ippo_SR': {'env.shared_rewards': True},
}

SEEDS = (1, 2, 3, 4, 5)
TOTAL_TIMESTEPS = 500_000_000
LOG_EVERY_STEPS = 2_000_000
FINAL_EVAL_EPISODES = 10_000
OUT_DIR = 'out/curves'
CHECKPOINT_DIR = 'out/checkpoints'
SAVE_CHECKPOINTS = True

EVAL_STALL_POLLS = 20


def run_path(condition, seed):
    return os.path.join(OUT_DIR, f'{ENV_NAME}_{condition}_s{seed}.jsonl')


def is_complete(path):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(line.startswith('{"kind": "done"') for line in f)


def script_sha256():
    with open(os.path.realpath(__file__), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def pufferlib_commit():
    import pufferlib
    repo = os.path.dirname(os.path.dirname(os.path.realpath(pufferlib.__file__)))
    try:
        out = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def apply_override(args, dotted, value):
    keys = dotted.split('.')
    node = args
    for key in keys[:-1]:
        assert key in node, f'unknown config section {key!r} in override {dotted!r}'
        node = node[key]
    assert keys[-1] in node, f'unknown config key {dotted!r}'
    node[keys[-1]] = value


def now_iso():
    return datetime.now(timezone.utc).isoformat()


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


def worker(condition, seed):
    sys.argv = sys.argv[:1]
    from pufferlib import pufferl

    args = pufferl.load_config(ENV_NAME)
    for dotted, value in CONDITIONS[condition].items():
        apply_override(args, dotted, value)

    args['seed'] = seed
    args['env']['rng'] = seed
    args['rank'] = 0
    args['world_size'] = 1
    args['gpu_id'] = 0
    args['nccl_id'] = b''
    args['wandb'] = False
    if TOTAL_TIMESTEPS is not None:
        args['train']['total_timesteps'] = TOTAL_TIMESTEPS

    pufferl.validate_config(args)
    total_timesteps = args['train']['total_timesteps']

    writer = Writer(run_path(condition, seed))
    writer.write('header', {
        'env_name': ENV_NAME,
        'condition': condition,
        'seed': seed,
        'hostname': socket.gethostname(),
        'started_at': now_iso(),
        'pufferlib_commit': pufferlib_commit(),
        'script_sha256': script_sha256(),
        'total_timesteps': total_timesteps,
        'log_every_steps': LOG_EVERY_STEPS,
        'args': args,
    })

    start = time.time()
    backend = pufferl._resolve_backend(args)
    p = None
    try:
        p = backend.create_pufferl(args)
        next_log_step = 0
        while p.global_step < total_timesteps:
            backend.rollouts(p)
            backend.train(p)
            if p.global_step < next_log_step:
                continue

            row = dict(pufferl.unroll_nested_dict(backend.log(p)))
            if 'env/episode_return' not in row:
                continue

            writer.write('row', row)
            next_log_step = p.global_step + LOG_EVERY_STEPS
            print(f'{condition} s{seed} {row["agent_steps"]:>14,} '
                f'return {row["env/episode_return"]:>10.3f} '
                f'sps {row.get("SPS", 0):>10,.0f}', flush=True)

        if SAVE_CHECKPOINTS:
            ckpt_dir = os.path.join(CHECKPOINT_DIR, f'{ENV_NAME}_{condition}_s{seed}')
            os.makedirs(ckpt_dir, exist_ok=True)
            backend.save_weights(p, os.path.join(ckpt_dir, f'{p.global_step:016d}.bin'))

        final_eval(backend, p, pufferl, writer)
        backend.close(p)
        writer.write('done', {
            'wall_time_s': time.time() - start,
            'final_step': p.global_step,
            'finished_at': now_iso(),
        })
    except Exception as e:
        writer.write('failed', {'error': repr(e), 'traceback': traceback.format_exc()})
        writer.close()
        raise

    writer.close()


def final_eval(backend, p, pufferl, writer):
    episodes = 0
    stalled = 0
    row = {}
    while episodes <= FINAL_EVAL_EPISODES:
        backend.rollouts(p)
        row = {**row, **dict(pufferl.unroll_nested_dict(backend.eval_log(p)))}
        n = row.get('env/n', 0)
        stalled = stalled + 1 if n <= episodes else 0
        episodes = max(episodes, n)
        if stalled >= EVAL_STALL_POLLS:
            print(f'WARNING: eval episode count stuck at {episodes}, aborting eval', flush=True)
            break

    writer.write('final_eval', row)


def driver(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
        help='Skip runs whose JSONL already has a done record')
    parser.add_argument('--only', type=str, default=None,
        help='Comma-separated subset of conditions')
    parser.add_argument('--seeds', type=str, default=None,
        help='Comma-separated subset of seeds')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('CONDITION', 'SEED'),
        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
        return 0

    conditions = args.only.split(',') if args.only else list(CONDITIONS)
    for c in conditions:
        assert c in CONDITIONS, f'unknown condition {c!r}'
        assert 'env_name' not in CONDITIONS[c], 'one build serves one env; drop the env_name override'
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(SEEDS)

    from pufferlib import pufferl
    backend = pufferl._resolve_backend({'env_name': ENV_NAME, 'slowly': False})
    cuda_build = hasattr(backend, 'create_pufferl')

    plan = list(itertools.product(conditions, seeds))
    skipped = {(c, s) for c, s in plan if args.resume and is_complete(run_path(c, s))}

    if args.dry_run:
        for condition, seed in plan:
            mark = 'skip' if (condition, seed) in skipped else 'write'
            print(f'{mark:>5}  {run_path(condition, seed)}')
        print(f'backend: {"ok" if cuda_build else f"CPU-only, rebuild with ./build.sh {ENV_NAME}"}')
        return 0

    assert cuda_build, f'backend has no create_pufferl: rebuild with ./build.sh {ENV_NAME}'

    os.makedirs(OUT_DIR, exist_ok=True)
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
        proc = subprocess.run([sys.executable, os.path.realpath(__file__),
            '--worker', condition, str(seed)])
        elapsed = time.time() - start
        durations.append(elapsed)
        results.append((condition, seed, 'ok' if proc.returncode == 0 else 'failed', elapsed))

    print('\n=== summary ===')
    for condition, seed, status, elapsed in results:
        print(f'{status:>8}  {condition:<12} s{seed}  {elapsed / 60:6.1f} min')

    return 1 if any(status == 'failed' for _, _, status, _ in results) else 0


if __name__ == '__main__':
    sys.exit(driver(sys.argv[1:]))

import argparse
import json
import os

import numpy as np

from harvest_env import HarvestEnv, HORIZON, CONFIG_PATH, read_config
from policies import make_agents

OUT_DIR = 'out'
JSON_PATH = os.path.join(OUT_DIR, 'schelling.json')
LOG_PATH = os.path.join(OUT_DIR, 'schelling.log')


def env_kwargs(config):
    return {k: v for k, v in config.get('env', {}).items() if k not in ('num_agents', 'rng')}


def run_episode(env, num_agents, cooperator_indices, policy_rng, thresholds, render=False):
    agents = make_agents(num_agents, cooperator_indices, *thresholds)
    returns = np.zeros(num_agents)
    for _ in range(HORIZON):
        env.step([agents[a].act(env.observations[a], policy_rng) for a in range(num_agents)])
        returns += env.rewards
        if render:
            env.render()
    return returns


def log_line(record):
    m = record['metrics']
    coop = record['cooperator_indices']
    returns = np.array(record['per_agent_returns'])
    mask = np.zeros(len(returns), dtype=bool)
    mask[coop] = True
    c_mean = returns[mask].mean() if mask.any() else float('nan')
    d_mean = returns[~mask].mean() if (~mask).any() else float('nan')
    return (f"j={record['j']} seed={record['seed']} ep={record['episode']} | "
            f"sust={m['sustainability']:.2f} equality={m['equality']:.3f} "
            f"depletion={m['time_to_depletion']:.0f} efficiency={m['efficiency']:.4f} | "
            f"C_return={c_mean:.1f} (n={mask.sum()}) D_return={d_mean:.1f} (n={(~mask).sum()})")


def thresholds(config):
    return (config['schelling']['cooperator_neighbor_threshold'],
            config['schelling']['defector_neighbor_threshold'])


def assignment(roles_rng, num_agents, j):
    return sorted(roles_rng.permutation(num_agents)[:j].tolist())


def replay(config, j, seed, episode):
    schelling = config['schelling']
    num_agents = schelling['num_agents']
    roles_rng = np.random.default_rng(seed)
    env = HarvestEnv(num_agents, seed, **env_kwargs(config))
    env.reset()
    try:
        # Episodes share one env stream, so earlier ones must be replayed to reach episode's state.
        for e in range(episode + 1):
            cooperator_indices = assignment(roles_rng, num_agents, j)
            returns = run_episode(env, num_agents, set(cooperator_indices),
                                  np.random.default_rng(seed), thresholds(config),
                                  render=(e == episode))
        record = {'j': j, 'seed': seed, 'episode': episode,
                  'cooperator_indices': cooperator_indices,
                  'per_agent_returns': returns.tolist(),
                  'metrics': env.log()}
    finally:
        env.close()
    print(log_line(record))


def sweep(config):
    schelling = config['schelling']
    num_agents = schelling['num_agents']
    kwargs = env_kwargs(config)
    episodes = []
    lines = []
    for j in range(num_agents + 1):
        for s in range(schelling['num_seeds']):
            seed = schelling['seed_base'] + s
            roles_rng = np.random.default_rng(seed)
            env = HarvestEnv(num_agents, seed, **kwargs)
            env.reset()
            try:
                for episode in range(schelling['episodes_per_seed']):
                    cooperator_indices = assignment(roles_rng, num_agents, j)
                    returns = run_episode(env, num_agents, set(cooperator_indices),
                                          np.random.default_rng(seed), thresholds(config))
                    record = {'j': j, 'seed': seed, 'episode': episode,
                              'cooperator_indices': cooperator_indices,
                              'per_agent_returns': returns.tolist(),
                              'metrics': env.log()}
                    episodes.append(record)
                    lines.append(log_line(record))
                    print(lines[-1], flush=True)
            finally:
                env.close()
    return episodes, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--render', action='store_true',
                        help='replay one episode on screen; writes nothing')
    parser.add_argument('--composition', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()

    config = read_config(args.config)
    if args.render:
        assert args.composition is not None and args.seed is not None, \
            '--render needs --composition and --seed'
        replay(config, args.composition, args.seed, args.episode)
        return

    episodes, lines = sweep(config)

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {'env_name': 'common_harvest',
               'config': {**config['schelling'],
                          'horizon': HORIZON,
                          'env': env_kwargs(config)},
               'episodes': episodes}
    with open(JSON_PATH, 'w') as f:
        json.dump(payload, f, indent=1)
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {JSON_PATH} ({len(episodes)} episodes) and {LOG_PATH}')


if __name__ == '__main__':
    main()

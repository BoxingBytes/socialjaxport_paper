import argparse
import ast
import configparser
import contextlib
import ctypes
import os

import numpy as np
import pufferlib
from pufferlib import _C

OBS_CHANNELS, OBS_WINDOW, NUM_ACTIONS, HORIZON = 2, 11, 8, 1000
TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT, NOOP, ZAP = range(NUM_ACTIONS)
PUFFERLIB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(pufferlib.__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def read_config(path=CONFIG_PATH):
    parser = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    parser.read(path)
    sections = {}
    for section in parser.sections():
        sections[section] = {}
        for key in parser[section]:
            try:
                sections[section][key] = ast.literal_eval(parser[section][key])
            except (ValueError, SyntaxError):
                sections[section][key] = parser[section][key]
    return sections


def _view(ptr, ctype, n):
    return np.ctypeslib.as_array((ctype * n).from_address(ptr))


@contextlib.contextmanager
def _pufferlib_cwd():
    previous = os.getcwd()
    os.chdir(PUFFERLIB_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


class HarvestEnv:
    def __init__(self, num_agents, seed, beam_blocks_movement=True,
                 differentiate_other_agents_in_obs=True, **env_kwargs):
        assert _C.env_name == 'common_harvest', f'_C holds {_C.env_name}'
        args = {
            'vec': {'total_agents': num_agents, 'num_buffers': 1, 'num_threads': 1},
            'env': {
                'num_agents': num_agents,
                'rng': seed,
                'beam_blocks_movement': beam_blocks_movement,
                'differentiate_other_agents_in_obs': differentiate_other_agents_in_obs,
                **env_kwargs,
            },
        }
        self.num_agents = num_agents
        self.seed = seed
        self.vec = _C.create_vec(args, 0)
        assert self.vec.obs_size == OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW
        assert self.vec.obs_dtype == 'ByteTensor'
        self.observations = _view(
            self.vec.obs_ptr, ctypes.c_uint8, num_agents*self.vec.obs_size
        ).reshape(num_agents, OBS_CHANNELS, OBS_WINDOW, OBS_WINDOW)
        self.rewards = _view(self.vec.rewards_ptr, ctypes.c_float, num_agents)
        self.terminals = _view(self.vec.terminals_ptr, ctypes.c_float, num_agents)
        self.actions = np.zeros((num_agents, self.vec.num_atns), dtype=np.float32)

    def reset(self):
        self.vec.reset()
        return self.observations

    def step(self, actions):
        self.actions[:, 0] = actions
        self.vec.cpu_step(self.actions.ctypes.data)
        return self.observations, self.rewards, self.terminals

    def render(self):
        with _pufferlib_cwd():
            self.vec.render(0)

    def log(self):
        return self.vec.log()

    def close(self):
        self.vec.close()
        self.observations = self.rewards = self.terminals = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num-agents', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=HORIZON)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = HarvestEnv(args.num_agents, args.seed)
    env.reset()
    returns = np.zeros(args.num_agents)
    try:
        for _ in range(args.steps):
            env.step(rng.integers(0, NUM_ACTIONS, size=args.num_agents))
            returns += env.rewards
            if args.render:
                env.render()
        print(f'returns {returns.tolist()} total {returns.sum():.1f}')
        print(f'log {env.log()}')
    finally:
        env.close()


if __name__ == '__main__':
    main()

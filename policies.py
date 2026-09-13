import argparse

import numpy as np

from harvest_env import (HarvestEnv, HORIZON, OBS_WINDOW, TURN_LEFT, FORWARD, BACKWARD,
                         STRAFE_LEFT, STRAFE_RIGHT, NOOP, read_config)

EMPTY, WALL, BEAM, APPLE, AGENT_BASE = 0, 1, 2, 3, 4
SELF_ROW, SELF_COL = 1, OBS_WINDOW//2  # verified against compute_observations, not the C comment
RESCAN_AFTER = 50

APPLE_MASK = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
              (-2, 0), (2, 0), (0, -2), (0, 2))
MOVEMENT_ACTIONS = (FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT)
NUM_DIRS = 4
STEP = {FORWARD: (1, 0), BACKWARD: (-1, 0), STRAFE_LEFT: (0, -1), STRAFE_RIGHT: (0, 1)}
ROW_ACTIONS = {1: FORWARD, -1: BACKWARD}
COL_ACTIONS = {1: STRAFE_RIGHT, -1: STRAFE_LEFT}


def live_neighbors(terrain, row, col):
    count = 0
    for dr, dc in APPLE_MASK:
        r, c = row + dr, col + dc
        if 0 <= r < OBS_WINDOW and 0 <= c < OBS_WINDOW and terrain[r, c] == APPLE:
            count += 1
    return count


def eligible_apples(terrain, min_neighbors):
    apples = np.argwhere(terrain == APPLE)
    if min_neighbors <= 0:
        return apples
    return np.array([(r, c) for r, c in apples
                     if live_neighbors(terrain, r, c) >= min_neighbors], dtype=int
                    ).reshape(-1, 2)


def as_cells(apples):
    return {(int(r), int(c)) for r, c in apples}


def eligible_set(terrain, min_neighbors):
    return as_cells(eligible_apples(terrain, min_neighbors))


def nearest_apple(apples):
    distances = np.abs(apples[:, 0] - SELF_ROW) + np.abs(apples[:, 1] - SELF_COL)
    order = np.lexsort((apples[:, 1], apples[:, 0], distances))
    return apples[order[0]]


# Moving onto an apple harvests it, so an apple below the threshold blocks like a wall.
def blocked(terrain, action, eligible):
    dr, dc = STEP[action]
    cell = (SELF_ROW + dr, SELF_COL + dc)
    if terrain[cell] in (WALL, AGENT_BASE):
        return True
    return terrain[cell] == APPLE and cell not in eligible


def step_towards(terrain, row, col, eligible):
    drow, dcol = row - SELF_ROW, col - SELF_COL
    axes = [(abs(drow), ROW_ACTIONS.get(np.sign(drow))),
            (abs(dcol), COL_ACTIONS.get(np.sign(dcol)))]
    for residual, action in sorted(axes, key=lambda axis: -axis[0]):
        if residual and not blocked(terrain, action, eligible):
            return action
    return None


def window_score(terrain, min_neighbors):
    return (len(eligible_apples(terrain, min_neighbors)),
            int((terrain == APPLE).sum()),
            int((terrain != WALL).sum()))


class ScriptedAgent:
    def __init__(self, min_neighbors):
        self.min_neighbors = min_neighbors
        self.heading = None
        self.start_scan()

    # The window shows 9 rows ahead and 1 behind, so a bad spawn heading is near-blind.
    def start_scan(self):
        self.scores = []
        self.pending_turns = 0

    def act(self, obs, rng):
        terrain = obs[0]
        if self.scores is not None:
            return self.scan(terrain)
        apples = eligible_apples(terrain, self.min_neighbors)
        eligible = as_cells(apples)
        if len(apples):
            action = step_towards(terrain, *nearest_apple(apples), eligible=eligible)
            if action is not None:
                self.heading = None
                self.fruitless = 0
                return action
        self.fruitless += 1
        if self.fruitless > RESCAN_AFTER:
            self.start_scan()
            return self.scan(terrain)
        return self.search(terrain, rng, eligible)

    # Four TURN_LEFTs sample every heading and return to the original one, then turn onto the best.
    def scan(self, terrain):
        if len(self.scores) < NUM_DIRS:
            self.scores.append(window_score(terrain, self.min_neighbors))
            if len(self.scores) == NUM_DIRS:
                self.pending_turns = int(np.argmax([  # np.argmax on tuples needs a flat key
                    s[0]*10000 + s[1]*100 + s[2] for s in self.scores]))
            return TURN_LEFT
        if self.pending_turns:
            self.pending_turns -= 1
            return TURN_LEFT
        self.scores = None
        self.heading = None
        self.fruitless = 0
        eligible = eligible_set(terrain, self.min_neighbors)
        return FORWARD if not blocked(terrain, FORWARD, eligible) else BACKWARD

    # Straight lines until obstructed: a diffusive walk barely leaves its spawn in HORIZON steps.
    def search(self, terrain, rng, eligible):
        if self.heading is None or blocked(terrain, self.heading, eligible):
            open_actions = [a for a in MOVEMENT_ACTIONS if not blocked(terrain, a, eligible)]
            if not open_actions:
                return NOOP  # boxed in by apples it refuses to harvest
            self.heading = open_actions[int(rng.integers(len(open_actions)))]
        return self.heading


def make_agents(num_agents, cooperator_indices, cooperator_threshold, defector_threshold):
    return [ScriptedAgent(cooperator_threshold if a in cooperator_indices else defector_threshold)
            for a in range(num_agents)]


def rollout(num_agents, seed, cooperator_indices, thresholds, steps=HORIZON, render=False):
    agents = make_agents(num_agents, cooperator_indices, *thresholds)
    rng = np.random.default_rng(seed)
    env = HarvestEnv(num_agents, seed)
    env.reset()
    returns = np.zeros(num_agents)
    try:
        for _ in range(steps):
            env.step([agents[a].act(env.observations[a], rng) for a in range(num_agents)])
            returns += env.rewards
            if render:
                env.render()
        return returns, env.log()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=7)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=HORIZON)
    parser.add_argument('--composition', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    schelling = read_config()['schelling']
    thresholds = (schelling['cooperator_neighbor_threshold'],
                  schelling['defector_neighbor_threshold'])
    compositions = ([args.composition] if args.composition is not None
                    else [0, args.num_agents])
    for j in compositions:
        returns, log = rollout(args.num_agents, args.seed, set(range(j)),
                               thresholds, args.steps, args.render)
        print(f'j={j} returns={returns.tolist()} total={returns.sum():.1f}')
        print(f'   log={ {k: round(v, 3) for k, v in log.items()} }')


if __name__ == '__main__':
    main()

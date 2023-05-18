import math
import random
from collections import defaultdict

import numpy as np
import numpy.ma as ma


class MCTree:
    def __init__(self, predict, skip, num_mcts_sims=2, cpuct=1.5) -> None:
        self._predict = predict
        self._skip = skip
        self._num_mcts_sims = num_mcts_sims
        self._cpuct = cpuct
        self._ps = {}
        self._ns = defaultdict(int)
        self._qsa = defaultdict(int)
        self._nsa = defaultdict(int)

    @staticmethod
    def _masked_array(array, indices):
        mask = np.ones(array.size, bool)
        mask[indices] = False
        return ma.masked_array(array, mask).filled(0)

    def _ucb(self, s, a):
        eps = random.uniform(1e-9, 1e-7) if self._nsa[(s, a)] == 0 else 0
        return self._qsa[(s, a)] + self._cpuct * self._ps[s][a] * (
            math.sqrt(self._ns[s] + eps) / (1 + self._nsa[(s, a)])
        )

    def _search(self, env):
        sa_records = []
        observation = env.observation
        reward = 0
        while True:
            s = observation.tobytes()
            actions = env.available_actions
            if s in self._ps:
                action_to_ucb = {a: self._ucb(s, a) for a in actions}
                a = max(action_to_ucb, key=action_to_ucb.get)
                sa_records.append((s, a))
                observation, reward, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    self._skip()
                    break
            else:
                p, v = self._predict(observation)
                p = self._masked_array(p, actions)
                if p.sum() == 0:
                    p[actions] = 1
                p /= p.sum()
                self._ps[s] = p
                reward = v
                break

        for s, a in reversed(sa_records):
            self._qsa[(s, a)] = (self._qsa[(s, a)] * self._nsa[(s, a)] + reward) / (
                self._nsa[(s, a)] + 1
            )
            self._nsa[(s, a)] += 1
            self._ns[s] += 1
            env.undo()

    def pi(self, env):
        for _ in range(self._num_mcts_sims):
            self._search(env)
        observation = env.observation
        s = observation.tobytes()
        n = np.array([self._nsa[(s, a)] for a in range(env.num_actions)])
        pi = n / self._ns[s]
        return pi

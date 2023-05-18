#use for confrontation training


import multiprocessing as mp

import numpy as np
from setproctitle import setproctitle

from learning.envs import env_cls
from learning.mcts import MCTree


class DuelWorker(mp.Process):
    def __init__(self, env_id, schedule_client, predict_client) -> None:
        super().__init__()
        self._env_id = env_id
        self._schedule_client = schedule_client
        self._predict_client = predict_client

    def run(self) -> None:
        setproctitle(f"duel-worker-{self.name.split('-')[-1]}")

        env = env_cls(self._env_id)()
        _, info = env.reset()
        mcts = [
            MCTree(self._predict_client.predict(model_name), self._predict_client.skip)
            for model_name in self._predict_client.model_names
        ]

        while True:
            num_steps = info["turns"]
            model_index = (self._schedule_client.episode_id + num_steps) % len(mcts)
            pi = mcts[model_index].pi(env)
            action = np.argmax(pi)
            _, reward, terminated, truncated, info = env.step(action)
            self._schedule_client.update(num_steps)
            if terminated or truncated:
                self._schedule_client.finish((model_index, reward))
                break

        self._predict_client.close()

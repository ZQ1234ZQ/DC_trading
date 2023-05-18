#单次运行跑一遍的结果，输出是最终收益结果与当前这一步获得的experience，reward在这完成了
#The result of a single run, the output is the final revenue result and the experience obtained in the current step, and the reward is completed here
import multiprocessing as mp
import os
import random

import numpy as np
from setproctitle import setproctitle

from learning.envs import env_cls
from learning.experience import Experience
from learning.mcts import MCTree


class ExecuteEpisodeWorker(mp.Process):
    def __init__(self, env_id, schedule_client, predict_client, choose_action) -> None:
        super().__init__()
        self._env_id = env_id
        self._schedule_client = schedule_client
        self._predict_client = predict_client
        self._model_name = predict_client.model_names[0]
        self._choose_action = choose_action

    def run(self) -> None:
        setproctitle(f"episode-{self.name.split('-')[-1]}")

        experiences = []
        env = env_cls(self._env_id)()
        state, _ = env.reset()
        predict = self._predict_client.predict(self._model_name)
        file1 = open("result", "w")  # append mode
        while True:
            q_values = predict(state)
            action = self._choose_action(env.available_actions, q_values)
            file1.write(f"{action}")
            next_state, reward, terminated, truncated, info = env.step(action)
            experiences.append(
                Experience(state, action, reward, next_state, terminated or truncated)
            )
            state = next_state
            self._schedule_client.update(info["turns"])
            if terminated or truncated:
                for portfolio in info["portfolios"]:
                    buy_price, money, next_event_type, _ = portfolio
                    if next_event_type == 1:
                        env._asset[1] -= money
                        env._asset[0] += money * (env._info["start_price"] / buy_price)
                    else:
                        env._asset[2] -= money
                        env._asset[0] += money * (buy_price / env._info["start_price"])
                for i in range(len(experiences) - 1, -1, -1):
                    experiences[i].reward = experiences[i].reward + env._asset[0]/1000000# final reward
                self._schedule_client.finish((experiences, env._asset[0] - 1000000))
                file1.close()
                break

        self._predict_client.close()

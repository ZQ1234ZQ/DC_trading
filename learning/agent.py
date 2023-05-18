#as the core to manage workers

import random

import numpy as np
import numpy.ma as ma

from learning.envs import env_cls
from learning.experience import ExperienceManager
from learning.utils import console
from learning.workers import (
    ExecuteEpisodeWorker,
    InitWorker,
    PredictWorker,
    ScheduleWorker,
    TrainWorker,
)


class Agent:
    def __init__(self, model_name, env_id) -> None:
        self._model_name = model_name
        self._env_id = env_id

        init_worker = InitWorker(self._model_name, self._env_id)
        init_worker.start()
        init_worker.join()

    def choose_action(self, epsilon=0):
        def _masked_array(array, indices):
            mask = np.ones(array.size, bool)
            mask[indices] = False
            return ma.masked_array(array, mask).filled(-np.inf)

        def _choose_action(actions, q_values):
            q_values = _masked_array(q_values, actions)
            if random.random() > epsilon:
                return np.argmax(q_values)
            else:
                return random.choice(actions)

        return _choose_action

    def exercise(self, num_episodes, epsilon):
        with PredictWorker([self._model_name]) as pw, ScheduleWorker(
            "The agent is playing game with itself to gain more experiences",
            num_episodes,
            env_cls(self._env_id).max_num_steps,
            [pw.ready],
        ) as sw:
            execute_episode_workers = []
            for i in range(num_episodes):
                execute_episode_worker = ExecuteEpisodeWorker(
                    self._env_id,
                    sw.register(i),
                    pw.register(),
                    self.choose_action(epsilon),
                )
                execute_episode_worker.start()
                execute_episode_workers.append(execute_episode_worker)

            for execute_episode_worker in execute_episode_workers:
                execute_episode_worker.join()

            new_experiences = []
            returns = []
            for result in sw.results:
                experience, return_ = result
                new_experiences.extend(experience)
                returns.append(return_)
            file1 = open("avg.txt", "a")  # append mode
            file1.write(f"{sum(returns) / len(returns)}\n")
            file1.close()

        experience_manager = ExperienceManager(self._model_name)
        experience_manager.append(new_experiences)

    def learn(self, new_model_name=None):
        console.log(
            "The agent is learning from its experiences to generate a new model."
        )

        new_model_name = new_model_name or f"{self._model_name}_new"

        train_worker = TrainWorker(self._model_name, new_model_name)
        train_worker.start()
        train_worker.join()

        return new_model_name

#env of DC trading strategy

import os
import random
from enum import Enum
from functools import cache

import numpy as np
import pandas as pd


class Action(Enum):
    NONE = 0
    BUY100 = 1
    BUY1000 = 2
    BUY10000 = 3
    BUY100000 = 4


class TradingEnv:
    _event_infos = pd.read_csv(os.path.join(os.path.dirname(__file__), "events.csv"))
    _events025 = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "EURGBP0.25.csv")
    ).drop(columns=["start_time", "end_time", "theta", "num"])
    _events050 = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "EURGBP0.5.csv")
    ).drop(columns=["start_time", "end_time", "theta", "num"])
    _events100 = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "EURGBP1.0.csv")
    ).drop(columns=["start_time", "end_time", "theta", "num"])
    _events150 = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "EURGBP1.5.csv")
    ).drop(columns=["start_time", "end_time", "theta", "num"])

    _duration = 500
    _num_actions = len(Action)
    _max_num_steps = len(_event_infos) - 1

    def __init__(self) -> None:
        self._begin_turn = random.randint(
            0, len(self._event_infos) - self._duration - 1
        )
        self._max_turns = self._begin_turn + self._duration
        self.diff = 1
        # self.diff = self._event_infos.iloc[self._begin_turn : self._max_turns + 1][
        #     "length"
        # ].sum()

    @classmethod
    @property
    def max_num_steps(self):
        return self._max_num_steps

    @classmethod
    @property
    def observation_shape(cls):
        return (5, 10, 6)

    @property
    def observation(self):
        event_info = self._event_infos.iloc[self._turns]
        i = int(event_info["0.25"])
        event025 = self._events025.iloc[i - 9 : i + 1].to_numpy()
        j = int(event_info["0.5"])
        event050 = self._events050.iloc[j - 9 : j + 1].to_numpy()
        k = int(event_info["1.0"])
        event100 = self._events100.iloc[k - 9 : k + 1].to_numpy()
        l = int(event_info["1.5"])
        event150 = self._events150.iloc[l - 9 : l + 1].to_numpy()
        portfolio = np.zeros((10, 6))
        for i in range(3):
            portfolio[:, i] = self._asset[i]
        portfolio[:, :3] = portfolio[:, :3] / 1000000
        return np.stack([event025, event050, event100, event150, portfolio])

    @property
    def _info(self):
        event_info = self._event_infos.iloc[self._turns]
        theta = str(event_info["theta"])
        event = {
            "0.25": self._events025,
            "0.5": self._events050,
            "1.0": self._events100,
            "1.5": self._events150,
        }[theta]
        event_index = int(event_info[theta])
        return {
            "turns": self._turns,
            "theta": float(theta),
            "event_type": event.iloc[event_index].event_type,
            "next_event_type": -event.iloc[event_index - 1].event_type,
            "start_price": event.iloc[event_index].start_price,
            "end_price": event.iloc[event_index].end_price,
            "length": event.iloc[event_index].length,
            "next_length": event.iloc[event_index + 1].length,
            "portfolios": self._portfolios,
        }

    def reset(self):
        self._turns = self._begin_turn
        self._asset = [1000000, 0, 0]
        self._portfolios = []
        self._records = []
        return self.observation, self._info

    @classmethod
    @property
    def num_actions(cls):
        return cls._num_actions

    @property
    def available_actions(self):
        if int(self._info["event_type"]) == 0:
            return [action.value for action in Action]
        else:
            return [Action.NONE.value]

    @property
    def _terminated(self):
        return self._turns >= self._max_turns

    @property
    def _truncated(self):
        return not self._terminated and self._turns >= self._max_num_steps

    def step(self, action):
        if action not in self.available_actions:
            print(f"Invalid action {action}.")
            print(f"Valid action as below: {self.available_actions}.")
            return

        stop_gain = self._info["theta"] * 0.01

        self._records.append((self._asset, self._portfolios.copy()))

        new_portfolios = []
        for portfolio in self._portfolios:
            end_price, money, next_event_type, target_price = portfolio
            if (
                next_event_type == 1
                and self._info["start_price"] >= target_price
                or next_event_type == -1
                and self._info["start_price"] <= target_price
            ):
                self._asset[0] += money * (self._info["start_price"] / end_price - 1)
                if next_event_type == 1:
                    self._asset[1] -= money
                    self._asset[0] += money * (self._info["start_price"] / end_price)
                else:
                    self._asset[2] -= money
                    self._asset[0] += money * (end_price / self._info["start_price"])
            else:
                new_portfolios.append(portfolio)
        self._portfolios = new_portfolios

        reward = 0
        if action != Action.NONE.value:
            money = 10 * 10**action

            if self._asset[0] >= money - 1000000:
                self._asset[0] -= money
                if self._info["next_event_type"] == 1:
                    self._asset[1] += money
                else:
                    self._asset[2] += money
                self._portfolios.append(
                    (
                        self._info["end_price"],
                        money,
                        self._info["next_event_type"],
                        self._info["end_price"] * (1 - stop_gain)
                        if self._info["next_event_type"] == -1
                        else self._info["end_price"] * (1 + stop_gain),
                    )
                )

                if self._info["next_length"] > stop_gain:
                    reward = 0.1  #瞬时奖励
        else:
            if self._info["next_length"] < stop_gain:
                reward = 0.1

        self._turns += 1
        return (
            self.observation,
            reward,
            self._terminated,
            self._truncated,
            self._info,
        )

    def undo(self):
        self._asset, self._portfolios = self._records.pop()
        self._turns -= 1

    def render(self):
        print(reward, self._info)


if __name__ == "__main__":
    env = TradingEnv()
    obs, info = env.reset()
    while True:
        print(env._portfolios)
        action = int(input())
        obs, reward, _, _, info = env.step(action)

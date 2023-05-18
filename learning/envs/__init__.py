from learning.envs.trading_v0 import TradingEnv


def env_cls(env_id):
    if env_id == "trading":
        return TradingEnv

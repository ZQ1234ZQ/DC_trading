import multiprocessing as mp

from learning.agent import Agent


def train(
    model_name,
    env_id,
    num_episodes=100,
    epsilon=1,
    epsilon_min=0.1,
    epsilon_decay=0.995,
):
    agent = Agent(model_name, env_id)
    for _ in range(num_episodes):
        agent.exercise(24, epsilon)
        agent.learn()
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

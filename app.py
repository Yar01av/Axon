"""
Run this file to use the application!
"""

import gym
from basic_dqn_agent import DQNAgent

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(8, 4, gym_env=env)
    agent.train(n_episodes=5)
    #agent.play(env)

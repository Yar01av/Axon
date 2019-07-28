"""
Run this file to use the application!
"""

import gym
from basic_dqn_agent import DQNAgent
from random_agent import RandomAgent

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')

    # Uncomment for DQN agent
    agent = DQNAgent(8, 4, gym_env=env)
    agent.train(n_episodes=100)
    agent.play()

    # Uncomment for random agent
    #agent = RandomAgent(env, 8, 4)
    #agent.play()

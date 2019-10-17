"""
Run this file to use the application!
"""

import gym
from basic_dqn_agent import DQNAgent
from env_config import lunar_lander_basic, cart_pole
from random_agent import RandomAgent

if __name__ == "__main__":
    config = lunar_lander_basic

    env = gym.make(config["env_name"])

    # Uncomment for DQN agent
    agent = DQNAgent(config["input_dim"], config["output_dim"], gym_env=env)
    agent.train(n_episodes=50)
    agent.play()

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

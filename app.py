"""
Run this file to use the application!
"""

import gym
from agents.keras_dqn_agent import KerasDQNAgent
from agents.torch_dqn_agent import TorchDQNAgent
from env_config import lunar_lander_basic, cart_pole
from agents.random_agent import RandomAgent

if __name__ == "__main__":
    config = lunar_lander_basic

    env = config["env"]

    # Uncomment for DQN agent
    agent = TorchDQNAgent(config["obs_dim"], config["action_dim"], gym_env=env)
    agent.train(n_episodes=400)
    agent.play()

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

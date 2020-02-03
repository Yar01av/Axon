"""
Run this file to use the application!
"""
from agents.dqn.fixed_dqn_agent import FixedDQNAgent
from agents.dqn.keras_dqn_agent import KerasDQNAgent
from env_config import lunar_lander_basic

if __name__ == "__main__":
    config = lunar_lander_basic

    env = config["env"]

    # Uncomment for DQN agent
    agent = FixedDQNAgent(config["obs_dim"], config["action_dim"], gym_env=env)
    agent.train(n_episodes=100)
    agent.play(n_episodes=5)

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

"""
Run this file to use the application!
"""
from agents.dqn.fixed_dqn_agent import FixedDQNAgent
from agents.dqn.k_steps_dqn import KStepsDQNAgent
from agents.dqn.keras_dqn_agent import KerasDQNAgent
from env_config import lunar_lander_basic

if __name__ == "__main__":
    config = lunar_lander_basic

    env = config["env"]

    # Uncomment for DQN agent
    agent = KStepsDQNAgent(config["obs_dim"], config["action_dim"], gym_env=env, k=4)
    agent.train(n_episodes=900)
    agent.play(n_episodes=5)

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

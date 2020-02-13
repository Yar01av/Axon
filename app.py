"""
Run this file to use the application!
"""
from operator import add

from tensorboardX import SummaryWriter
import numpy as np
from agents.agent import Agent
from agents.dqn.fixed_dqn_agent import FixedDQNAgent
from agents.dqn.k_steps_dqn import KStepsDQNAgent
from agents.dqn.keras_dqn_agent import KerasDQNAgent
from env_config import lunar_lander_basic
from other.analytics import Logger
from other.util import Variable

# Prepare some extra callbacks for analytics
plotter = SummaryWriter(comment="xFixedDQN")
episode_reward_sum = Variable(0)
logger = Logger()
analytical_callbacks = Agent.Callbacks(
    after_step_cbs=[lambda s: episode_reward_sum.modify_value(add, episode_reward_sum.get_value(), s["reward"]),
                    lambda s: logger.remember(s["reward"])],
    after_episode_cbs=[lambda s: plotter.add_scalar("Per Episode Reward",
                                                    episode_reward_sum.get_value()
                                                    , s["e"]),
                       lambda s: episode_reward_sum.modify_value(lambda: 0),
                       lambda s: logger.log(
                           f"Score: {np.sum(logger.memory)} \t Episode: {s['e']}"),
                       lambda s: logger.forget()],
    after_gameloop_cbs=[
        lambda s: plotter.close()
    ])


if __name__ == "__main__":
    config = lunar_lander_basic

    env = config["env"]

    # Uncomment for DQN agent
    agent = FixedDQNAgent(config["obs_dim"], config["action_dim"], gym_env=env)
    agent.train(n_episodes=100, extra_callbacks=analytical_callbacks)
    agent.play(n_episodes=5)

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

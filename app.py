"""
Run this file to use the application!
"""
from operator import add

from tensorboardX import SummaryWriter
import numpy as np

from agents.PG.a2c_agent import A2CAgent
from agents.PG.reinforce_agent import REINFORCEAgent
from agents.agent import Agent
from agents.DQN.double_dgn_agent import DoubleDQNAgent
from agents.DQN.fixed_dqn_agent import FixedDQNAgent
from agents.DQN.k_steps_dqn import KStepsDQNAgent
from agents.DQN.keras_dqn_agent import KerasDQNAgent
from env_config import lunar_lander_basic, cart_pole
from other.analytics import Logger
from other.util import Variable
from numpy.random import seed
from tensorflow import set_random_seed


# Seed to make sure that the results are reproducible
#seed(1)
#set_random_seed(123)

# Prepare some extra callbacks for analytics
def _get_analytical_callbacks():
    plotter = SummaryWriter(comment="xREINFORCE")
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

    return analytical_callbacks


# Returns callbacks that handle loading and saving while the agent is training or playing.
# Make sure that the agent actually has a model. Furthermore, make sure that the model used by the agent actually
# implements Model base class. This is, sadly, not always the case but should be.
def _get_storage_callbacks(agent_instance, save_path=None, load_path=None):
    if save_path is not None:
        after_gameloop_cbs = [lambda s: agent_instance._model.save(save_path)]
    else:
        after_gameloop_cbs = None

    if load_path is not None:
        pre_gameloop_cbs = [lambda s: agent_instance._model.load_model(load_path)]
    else:
        pre_gameloop_cbs = None

    return Agent.Callbacks(pre_gameloop_cbs=pre_gameloop_cbs, after_gameloop_cbs=after_gameloop_cbs)


if __name__ == "__main__":
    config = cart_pole

    env = config["env"]

    # Warm up
    agent = REINFORCEAgent(env, config["obs_dim"], config["action_dim"])
    agent.train(n_episodes=10, extra_callbacks=_get_analytical_callbacks())

    # Do several runs to negate unlucky initializations
    for i in range(5):
        agent = REINFORCEAgent(env, config["obs_dim"], config["action_dim"])
        agent.train(n_episodes=1000,
                    extra_callbacks=_get_analytical_callbacks())
                                    #+_get_storage_callbacks(agent, save_path="last_save.h5"))

    agent.play(n_episodes=10)
               #, extra_callbacks=_get_storage_callbacks(agent, load_path="last_save.h5"))

    # Uncomment for random agent
    #agent = RandomAgent(env, config["input_dim"], config["output_dim"])
    #agent.play()

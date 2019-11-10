# Basic configurations of the possible environments
import gym

lunar_lander_basic = {"env": gym.make('LunarLander-v2'),
                      "obs_dim": 8,
                      "action_dim": 4}

cart_pole = {"env": gym.make('CartPole-v1'),
             "obs_dim": 4,
             "action_dim": 2}
import gym
from abc import ABC, abstractmethod
import numpy as np

from other.util import execute_callbacks


class Agent(ABC):
    """
    Fundamental class that contains the game loops
    """
    def __init__(self, gym_env, state_size):
        self.state_size = state_size
        self.gym_env = gym_env

    @abstractmethod
    def act(self, state):
        pass

    def _prep_fresh_state(self, gym_env):
        """
        Create a new start state

        :param gym_env:
        :return:
        """

        state = gym_env.reset()
        return np.reshape(state, [1, self.state_size])

    def _play_through(self, max_episode_length=3000, n_episodes=200, after_step_callbacks=None, after_gameloop_callbacks=None,
                      after_episode_callbacks=None):
        """
        Go through the main game loop (e.g. for training or just playing)

        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param after_episode_callbacks: methods to execute after the end of an episode
        :param after_step_callbacks: methods to execute after the end of a step
        :param after_gameloop_callbacks: methods to execute after the end of a game (episode)
        :return: None
        """

        for e in range(n_episodes):
            # reset state in the beginning of each game
            state = self._prep_fresh_state(self.gym_env)
            com_reward = 0  # total reward per episode
            # time_t represents each frame of the game
            # the more time_t the more score
            for time_t in range(max_episode_length):
                # Decide on an action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                next_state, reward, done, _ = self.gym_env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # Increase the total reward
                com_reward += reward

                execute_callbacks(after_step_callbacks, locals())

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True, then the game ends
                if done or time_t == max_episode_length-1:
                    # break out of the loop
                    break

            execute_callbacks(after_episode_callbacks, locals())

        execute_callbacks(after_gameloop_callbacks, locals())

from abc import ABC, abstractmethod
import numpy as np
from other.util import execute_callbacks, safe_list_assignment


class Agent(ABC):
    """
    Fundamental class that contains the game loops. Discrete observations and actions seen as a choice from a
    finite array are assumed.
    """
    class Callbacks:
        """
        Class that encapsulates actions to be taken at different stages of the gameplay loop
        """
        @abstractmethod
        def __init__(self, after_step_cbs=None, after_gameloop_cbs=None, after_episode_cbs=None):
            """
            :param after_step_cbs: List of callbacks to be executed after each step
            :param after_gameloop_cbs: List of callbacks to be executed at the end of all the games
            :param after_episode_cbs: List of callbacks to be executed after each game
            """

            self.after_step_cbs = safe_list_assignment(after_step_cbs)
            self.after_gameloop_cbs = safe_list_assignment(after_gameloop_cbs)
            self.after_episode_cbs = safe_list_assignment(after_episode_cbs)

        def add_after_step_callback(self, cb):
            self.after_step_cbs.append(cb)

        def after_gameloop_callback(self, cb):
            self.after_gameloop_cbs.append(cb)

        def after_episode_callback(self, cb):
            self.after_episode_cbs.append(cb)

    @abstractmethod
    def __init__(self, gym_env, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.gym_env = gym_env

    @abstractmethod
    def _play_callbacks_factory(self) -> Callbacks:
        """
        Override this to use callbacks when playing

        :return: A group of play callbacks as Callbacks object
        """

        pass

    @abstractmethod
    def _train_callbacks_factory(self) -> Callbacks:
        """
        Override this to use callbacks when training

        :return: A group of train callbacks as Callbacks object
        """
        pass

    @abstractmethod
    def act(self, state):
        pass

    def play(self, max_episode_length=3000, n_episodes=200):
        """
        Plays the game using the agent and is a curried method

        :param max_episode_length: maximum length of each game
        :param n_episodes: number of games to play
        :return: None
        """

        self._play_through(max_episode_length, n_episodes, callbacks="play")

    def train(self, max_episode_length=3000, n_episodes=200):
        """
        Trains the agent

        :param max_episode_length: maximum length of each game
        :param n_episodes: number of games to play
        :return: None
        """

        self._play_through(max_episode_length, n_episodes, callbacks="train")

    def _prep_fresh_state(self, gym_env):
        """
        Create a new start state

        :param gym_env:
        :return:
        """

        state = gym_env.reset()
        return np.reshape(state, [1, self.state_size])

    def _play_through(self, max_episode_length=3000, n_episodes=200, callbacks="play"):
        """
        Go through the main game loop (e.g. for training or just playing)

        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param callbacks: Callbacks to be executed at different stages in the gameloop. "play" uses the instance
        returned by _play_callbacks_factory and "train" uses the instance
        returned by _train_callbacks_factory
        :return: None
        """

        # Translate
        if callbacks == "play":
            callbacks = self._play_callbacks_factory()
        elif callbacks == "train":
            callbacks = self._train_callbacks_factory()
        else:
            raise AssertionError("Invalid choice for callbacks argument!")

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

                execute_callbacks(callbacks.after_step_cbs, locals())

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True, then the game ends
                if done or time_t == max_episode_length-1:
                    # break out of the loop
                    break

            execute_callbacks(callbacks.after_episode_cbs, locals())

        execute_callbacks(callbacks.after_gameloop_cbs, locals())

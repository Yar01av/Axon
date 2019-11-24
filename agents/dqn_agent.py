from abc import abstractmethod
import numpy as np
from tensorboardX import SummaryWriter

from agents.agent import Agent
from models.model import Model
from other.util import get_timeseq_diff
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam
from random import random
import random
from keras.models import load_model
from datetime import datetime
from other.analytical_engine import AggregPlotter, Logger


# Deep Q-learning Agent
class DQNAgent(Agent):
    # Note that the learning rate is only considered if there is no model provided
    @abstractmethod
    def __init__(self, state_size, action_size, gym_env, memory=deque(maxlen=1000000), gamma=0.99, batch_size=34,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.996, learning_rate=0.001, model: Model = None):
        super().__init__(gym_env, state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.explore = False
        self._model = model

        if model is None:
            self._model = self._build_model()

    def _play_callbacks_factory(self):
        return Agent.Callbacks(after_step_cbs=[lambda s: self.gym_env.render()])

    def _train_callbacks_factory(self):
        # Set up an analytical tools
        plotter = SummaryWriter(comment="Rewards per Episode")
        logger = Logger()

        # Prepare callbacks
        after_step_callbacks = [lambda s: self.remember(s["state"], s["action"], s["reward"], s["next_state"],
                                                        s["done"]),  # Store the experience
                                lambda s: self._replay(self.batch_size),  # Train on the experiences
                                lambda s: self._explore_less(),
                                # Do analytics
                                lambda s: s.update({"episode_reward_sum":
                                                    s.get("episode_reward_sum", 0) + s["reward"]}),
                                lambda s: logger.remember(s["reward"])]  # Remember for future logging
        after_episode_callbacks = [lambda s: plotter.add_scalar("Per Episode Reward", s["episode_reward_sum"],
                                                                s["e"]),
                                   lambda s: s.update({"episode_reward_sum": 0}),
                                   lambda s: logger.log(f"Score: {np.sum(logger.memory)} \t Episode: {s['e']}"),
                                   lambda s: logger.forget()]  # Empty the memory after taking the sum
        after_gameloop_callbacks = [lambda s: plotter.close()]
        # plot the lengths of the games (differences of each sequence)

        return Agent.Callbacks(after_step_cbs=after_step_callbacks,
                               after_episode_cbs=after_episode_callbacks,
                               after_gameloop_cbs=after_gameloop_callbacks)

    # Neural Net for Deep-Q learning Model
    @abstractmethod
    def _build_model(self):
        pass

    def _explore_less(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _dqn_play(self, n_episodes=100, max_episode_length=3000, save_path="last_save.h5",
                  explore=True, load_path=None, callbacks="play"):
        """
        Go through the main game loop (e.g. for training or just playing) using
        DQN related features (like saving models and exploration)

        :param load_path: load the _model from...
        :param batch_size: number of experiences to consider at every training
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param save_path: where to save the network after training
        :param explore: randomize the action for exploration when true
        :return: None
        """
        # Overwrite the build _model if needed
        if load_path is not None:
            self._model.load_model(load_path)

        self.explore = explore  # Explore if needed

        # Iterate the game
        super()._play_through(max_episode_length=max_episode_length, n_episodes=n_episodes, callbacks=callbacks)

        if save_path is not None:
            self.save_as(save_path)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Randomize the action for exploration
        if np.random.rand() <= self.epsilon and self.explore:
            return random.randrange(self.action_size)

        act_values = self._model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def save_as(self, path):
        self._model.save(path)

    @abstractmethod
    def _replay(self, batch_size):
        """
        Learn from experiences

        :param batch_size: How many randomly chosen experiences from the memory to use
        :return: None
        """

        pass

    def train(self, batch_size=64, n_episodes=100, max_episode_length=3000, save_path="last_save.h5",
              load_path=None):
        """
        Trains the agent by playing games

        :param load_path: load the _model from...
        :param batch_size: number of experiences to consider at every training
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param save_path: where to save the network after training
        :return: None
        """

        self._dqn_play(n_episodes=n_episodes, max_episode_length=max_episode_length, save_path=save_path,
                       load_path=load_path, callbacks="train")

    def play(self, n_episodes=100, max_episode_length=3000, load_path="last_save.h5"):
        """
        Plays games without training

        :param load_path: load the _model from...
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :return: None
        """

        self._dqn_play(n_episodes=n_episodes, max_episode_length=max_episode_length, load_path=load_path,
                       explore=False)

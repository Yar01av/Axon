from abc import abstractmethod
import numpy as np
from tensorboardX import SummaryWriter
from operator import add
from agents.agent import Agent
from models.model import Model
from other.util import get_timeseq_diff, Variable
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam
from random import random
import random
from keras.models import load_model
from datetime import datetime
from other.analytics import Logger


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
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self._explore = Variable(False)
        self._model = model

        if model is None:
            self._model = self._build_model()

    def _play_callbacks_factory(self):
        return Agent.Callbacks(after_step_cbs=[lambda s: self.gym_env.render()])

    def _train_callbacks_factory(self):
        # Prepare callbacks
        after_step_callbacks = [lambda s: self._remember(s["state"], s["action"], s["reward"], s["next_state"],
                                                         s["done"], s),  # Store the experience
                                lambda s: self._replay(self.batch_size, s),  # Train on the experiences
                                lambda s: self._explore_less()]
        # plot the lengths of the games (differences of each sequence)

        return Agent.Callbacks(pre_gameloop_cbs=[lambda s: self._explore.modify_value(lambda: True)],
                               after_step_cbs=after_step_callbacks)

    # Neural Net for Deep-Q learning Model
    @abstractmethod
    def _build_model(self):
        pass

    def _explore_less(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _remember(self, state, action, reward, next_state, done, gameplay_state):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Randomize the action for exploration
        if np.random.rand() <= self.epsilon and self._explore.value:
            return random.randrange(self.action_size)

        act_values = self._model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def save_as(self, path):
        self._model.save(path)

    @abstractmethod
    def _replay(self, batch_size, gameplay_state):
        """
        Learn from experiences

        :param batch_size: How many randomly chosen experiences from the memory to use
        :param gameplay_state: Current state of the gameplay (e.g. which episode is it)
        :return: None
        """

        pass

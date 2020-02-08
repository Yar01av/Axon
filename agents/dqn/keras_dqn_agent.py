import random
from collections import deque
import numpy as np
from keras import Sequential
from keras.layers import Dense
from agents.dqn.dqn_agent import DQNAgent
from models.keras_model import KerasModel


class KerasDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, gym_env, memory=deque(maxlen=1000000), gamma=0.99, batch_size=34,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.996, learning_rate=0.001, model=None):
        super().__init__(state_size, action_size, gym_env, memory, gamma, batch_size, epsilon, epsilon_min,
                         epsilon_decay, learning_rate, model)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        return KerasModel(model, lr=self.learning_rate)

    def _replay(self, batch_size, gameplay_state):
        """
        Learn from experiences

        :param batch_size: How many randomly chosen experiences from the memory to use
        :return: None
        """

        if len(self.memory) < batch_size:
            return

        actions, dones, next_states, rewards, states = self._extract_minibatch(batch_size)

        targets = rewards + self.gamma * (np.amax(self._model.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self._model.predict(states)
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self._model.fit(states, targets_full)

    def _extract_minibatch(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        return actions, dones, next_states, rewards, states

from collections import deque

import numpy as np

from agents.DQN.keras_dqn_agent import KerasDQNAgent
from models.model import Model


class FixedDQNAgent(KerasDQNAgent):
    def __init__(self, state_size, action_size, gym_env, memory=deque(maxlen=1000000), gamma=0.99, batch_size=34,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.996, learning_rate=0.001, model: Model = None):
        super().__init__(state_size, action_size, gym_env, memory, gamma, batch_size,
                         epsilon, epsilon_min, epsilon_decay, learning_rate, model)
        self._target_model = self._build_model()

    def _train_callbacks_factory(self):
        def swapper(game_state):
            self._target_model = self._model

        original_callbacks = super()._train_callbacks_factory()
        original_callbacks.add_after_episode_callback(swapper)

        return original_callbacks

    def _replay(self, batch_size, gameplay_state):
        if len(self.memory) < batch_size:
            return

        actions, dones, next_states, rewards, states = self._extract_minibatch(batch_size)

        targets = rewards + self.gamma * (np.amax(self._target_model.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self._model.predict(states)
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self._model.fit(states, targets_full)
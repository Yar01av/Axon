from collections import deque

import numpy as np

from agents.dqn.k_steps_dqn import KStepsDQNAgent
from models.model import Model


class DoubleDQNAgent(KStepsDQNAgent):
    """
    This agent extends FixedDQNAgent and introduces k step jumps whereby k consequtive steps are merged into one by
    unfolding Bellman equation.
    """

    def __init__(self, state_size, action_size, gym_env, memory=deque(maxlen=1000000), gamma=0.99, batch_size=34,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.996, learning_rate=0.001, model: Model = None, k=1):

        super().__init__(state_size, action_size, gym_env, memory, gamma, batch_size,
                         epsilon, epsilon_min, epsilon_decay, learning_rate, model, k)

    def _replay(self, batch_size, gameplay_state):
        # TODO: this overriding breaks ehat is inherited from k-steps dqn
        if len(self.memory) < batch_size:
            return

        actions, dones, next_states, rewards, states = self._extract_minibatch(batch_size)
        ind = np.array([i for i in range(batch_size)])
        target_model_predictions = self._target_model.predict(next_states)
        primary_model_predictions = self._model.predict(next_states)

        # Let the primary model pick the action in the next state but ask the target model for its q-value
        targets = rewards + self.gamma * \
                  (target_model_predictions[[ind], [np.argmax(primary_model_predictions, axis=1)]]) * \
                  (1 - dones)
        targets_full = self._model.predict(states)
        targets_full[[ind], [actions]] = targets

        self._model.fit(states, targets_full)
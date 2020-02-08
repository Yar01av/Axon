from collections import deque

from agents.dqn.fixed_dqn_agent import FixedDQNAgent
from models.model import Model


class KStepsDQNAgent(FixedDQNAgent):
    """
    This agent extends FixedDQNAgent and introduces k step jumps whereby k consequtive steps are merged into one by
    unfolding Bellman equation.
    """

    def __init__(self, state_size, action_size, gym_env, memory=deque(maxlen=1000000), gamma=0.99, batch_size=34,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.996, learning_rate=0.001, model: Model = None, k=1):
        assert k >= 1

        super().__init__(state_size, action_size, gym_env, memory, gamma, batch_size,
                         epsilon, epsilon_min, epsilon_decay, learning_rate, model)
        self._k = k
        self._accumulated_reward = 0

    def _replay(self, batch_size, gameplay_state):
        """
        :param kwargs (time_t): the current step number within the episode (starting with 0)
        """

        if gameplay_state["time_t"] % self._k != 0:
            pass
        else:
            super()._replay(batch_size, gameplay_state)

    def _remember(self, state, action,  reward, next_state, done, gameplay_state):
        # Differentiate between k-th steps and the other steps
        if (gameplay_state["time_t"]+1) % self._k != 0:
            self._accumulated_reward += reward*self.gamma**(gameplay_state["time_t"] % self._k)
        else:
            self._accumulated_reward += reward*self.gamma**(self._k-1)
            super()._remember(state, action, self._accumulated_reward, next_state, done, gameplay_state)
            self._accumulated_reward = 0

    # TODO test
from typing import Any

import numpy as np
from torch import cuda, nn, optim, FloatTensor, LongTensor

from agents.PG.reinforce_agent import REINFORCEAgent
import torch.nn.functional as F
from agents.PG.util.memory_util import CompositeMemory
from agents.agent import Agent


class SimpleA2CNetwork(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )

        self._value_head = nn.Sequential(
            nn.Linear(128, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        base_output = self._base(x)
        return self._policy_head(base_output), self._value_head(base_output)


class A2CAgent(Agent):
    def __init__(self, gym_env, state_size, action_size, n_episodes_to_train=4, learning_rate=0.01, gamma=0.99,
                 model=None):
        super().__init__(gym_env, state_size, action_size)

        self._gamma = gamma
        self._lr = learning_rate
        self._n_episodes_to_train = n_episodes_to_train
        self._memory = CompositeMemory()

        if model is None:
            self._model = self._build_model()

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

    def _play_callbacks_factory(self):
        return Agent.Callbacks(after_step_cbs=[lambda s: self.gym_env.render()])

    def _train_callbacks_factory(self):
        return Agent.Callbacks(after_step_cbs=[lambda s: self._remember(s["state"], s["action"], s["reward"],
                                                                        s["done"])],
                               after_episode_cbs=[lambda s: self._learn(s)])

    def act(self, state):
        probs = self._model(FloatTensor(state).cuda())[0].softmax(dim=1)

        return np.random.choice(range(self.action_size), p=np.squeeze(probs.cpu().detach().numpy()))

    def _build_model(self):
        cuda.set_device(0)

        model = SimpleA2CNetwork(self.state_size, self.action_size)

        return model.cuda()

    def _remember(self, state, action, reward, done):
        if not done:
            self._memory.actions.append(action)
            self._memory.states.append(state)
            self._memory.rewards.append(reward)
        else:
            self._memory.q_vals.extend(self._memory.compute_qvals(self._gamma))

    def _learn(self, engine_state):
        if (engine_state["e"]+1) % self._n_episodes_to_train == 0:
            t_act = LongTensor(self._memory.actions).cuda()
            t_state = FloatTensor(np.squeeze(self._memory.states)).cuda()
            t_qval = FloatTensor(self._memory.q_vals).cuda()

            self._optimizer.zero_grad()
            t_policy = self._model(t_state)[0]
            t_advantage = t_qval - (self._model(t_state)[1]).detach()

            # Calculate the policy gradient replacing the Q-values by Advantage
            policy_loss = -(t_advantage*t_policy.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()
            # Calculate the value prediction loss using the definition of state value with respect to Q-values
            input = self._model(t_state)[1]
            #print(t_qval)
            value_loss = F.mse_loss(t_qval, input.squeeze(-1))
            #print(value_loss)

            (policy_loss + value_loss).backward()
            self._optimizer.step()

            # The training step is done. Prepare for the new data.
            self._memory.reset()
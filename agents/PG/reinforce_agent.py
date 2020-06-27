import numpy as np
from torch import nn, cuda, FloatTensor, ByteTensor, optim, log_softmax, LongTensor, from_numpy

from agents.PG.util.memory_util import CompositeMemory
from agents.agent import Agent
from models.torch_model import TorchModel


class REINFORCEAgent(Agent):
    """
    Agent that follows PG algorithm. This implementation is build specifically with PyTorch and cannot be easily
    switched to Keras.
    """

    #TODO:
    # > The model used is used without the wrapper as it was too restrictive. Change the code to make it use the wrapper

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
        probs = self._model(FloatTensor(state).cuda()).softmax(dim=1)

        return np.random.choice(range(self.action_size), p=np.squeeze(probs.cpu().detach().numpy()))

    def _build_model(self):
        cuda.set_device(0)

        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

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
            t_policy = self._model(t_state)

            # Compute the policy gradient
            loss = -(t_qval*t_policy.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()

            loss.backward()
            self._optimizer.step()

            # The training step is done. Prepare for the new data.
            self._memory.reset()
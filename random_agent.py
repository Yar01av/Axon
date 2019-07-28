import random

from agent import Agent


class RandomAgent(Agent):
    def _play_callbacks_factory(self) -> Agent.Callbacks:
        return Agent.Callbacks(after_step_cbs=[lambda s: self.gym_env.render()])

    def _train_callbacks_factory(self) -> Agent.Callbacks:
        return Agent.Callbacks()

    def __init__(self, gym_env, state_size, action_size):
        super().__init__(gym_env, state_size, action_size)

    def play(self, max_episode_length=3000, n_episodes=200):
        super()._play_through(max_episode_length, n_episodes)

    def act(self, state):
        return random.choice(range(self.action_size))

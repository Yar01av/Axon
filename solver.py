import datetime
# TODO: this timing should be replaced by the analytical tool
print(datetime.datetime.now())

import gym
import numpy as np
from util import execute_callbacks
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam
from random import random
import random
from keras.models import load_model


# EPISODES = 100
# MAX_EPISODE_LENGTH = 3000
# REPLAY_SIZE = 64
# train_flag = True  # Change this to train or see the results
# DEFAULT_SAVE_PATH = "last_save3.h5"
printing_resolution = 2


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _prep_fresh_state(self, gym_env):
        """
        Create a new start state

        :param gym_env:
        :return:
        """

        state = gym_env.reset()
        return np.reshape(state, [1, self.state_size])

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def _explore_less(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _play_through(self, gym_env, batch_size=64, n_episodes=100, max_episode_length=3000, save_path="last_save.h5",
                      load_path=None, after_step_callbacks=None, after_game_callbacks=None,
                      after_episode_callbacks=None):
        """
        Go through the main game loop (e.g. for training or just playing)

        :param load_path: load the model from...
        :param batch_size: number of experiences to consider at every training
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param save_path: where to save the network after training
        :param gym_env: the game environment
        :param after_episode_callbacks: methods to execute after the end of an episode
        :param after_step_callbacks: methods to execute after the end of a step
        :param after_game_callbacks: methods to execute after the end of a game (episode)
        :return: None
        """
        # Iterate the game
        for e in range(n_episodes):
            # Overwrite the build model if needed
            if load_path is not None:
                self.model = load_model(load_path)

            # reset state in the beginning of each game
            state = self._prep_fresh_state(gym_env)
            com_reward = 0  # total reward per episode
            # time_t represents each frame of the game
            # the more time_t the more score
            for time_t in range(max_episode_length):
                # Decide on an action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                next_state, reward, done, _ = gym_env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # Increase the total reward
                com_reward += reward

                execute_callbacks(after_step_callbacks, locals())

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True, then the game ends
                if done or time_t == max_episode_length-1:
                    # print the score
                    if (e % printing_resolution) == 0:
                        print("episode: {}/{}, score: {}".format(e, n_episodes, com_reward))

                    # break out of the loop
                    break

            execute_callbacks(after_episode_callbacks, locals())

        if save_path is not None:
            self.save_as(save_path)

        execute_callbacks(after_game_callbacks, locals())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def save_as(self, path):
        self.model.save(path)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict(states)
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

    def train(self, gym_env, batch_size=64, n_episodes=100, max_episode_length=3000, save_path="last_save.h5",
              load_path=None):
        """
        Trains the agent by playing games

        :param load_path: load the model from...
        :param batch_size: number of experiences to consider at every training
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param save_path: where to save the network after training
        :param gym_env: the game environment
        :return: None
        """

        after_step_callbacks = [lambda s: self.remember(s["state"], s["action"], s["reward"], s["next_state"],
                                                           s["done"]),
                                lambda s: self.replay(batch_size),
                                lambda s: self._explore_less()]
        self._play_through(gym_env, n_episodes=n_episodes, max_episode_length=max_episode_length, save_path=save_path,
                           load_path=load_path, after_step_callbacks=after_step_callbacks)

    def play(self, gym_env, n_episodes=100, max_episode_length=3000, load_path="last_save.h5"):
        """
        Plays games without training

        :param load_path: load the model from...
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param gym_env: the game environment
        :return: None
        """

        self._play_through(gym_env, n_episodes=n_episodes, max_episode_length=max_episode_length, load_path=load_path,
                           after_step_callbacks=[lambda s: gym_env.render()])


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(8, 4)
    #agent.train(env, n_episodes=400)
    agent.play(env)





print(datetime.datetime.now())
import gym
import numpy as np
from other.util import execute_callbacks, get_timeseq_diff
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam
from random import random
import random
from keras.models import load_model
from datetime import datetime
from other.analytical_engine import AggregPlotter, Logger

# EPISODES = 100
# MAX_EPISODE_LENGTH = 3000
# REPLAY_SIZE = 64
# DEFAULT_SAVE_PATH = "last_save3.h5"


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, memory=deque(maxlen=1000000), gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.996, learning_rate=0.001, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        if model is None:
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
                      explore=True, load_path=None, after_step_callbacks=None, after_gameloop_callbacks=None,
                      after_episode_callbacks=None):
        """
        Go through the main game loop (e.g. for training or just playing)

        :param load_path: load the model from...
        :param batch_size: number of experiences to consider at every training
        :param max_episode_length: length of each game
        :param n_episodes: number of games
        :param save_path: where to save the network after training
        :param gym_env: the game environment
        :param explore: randomize the action for exploration when true
        :param after_episode_callbacks: methods to execute after the end of an episode
        :param after_step_callbacks: methods to execute after the end of a step
        :param after_gameloop_callbacks: methods to execute after the end of a game (episode)
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
                    # break out of the loop
                    break

            execute_callbacks(after_episode_callbacks, locals())

        if save_path is not None:
            self.save_as(save_path)

        execute_callbacks(after_gameloop_callbacks, locals())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore=False):
        # Randomize the action for exploration
        if np.random.rand() <= self.epsilon and explore:
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

        # Set up an analytical tools
        plotter = AggregPlotter()
        logger = Logger()

        # Prepare callbacks
        after_step_callbacks = [lambda s: self.remember(s["state"], s["action"], s["reward"], s["next_state"],
                                                           s["done"]),  # Store the experience
                                lambda s: self.replay(batch_size),  # Train on the experiences
                                lambda s: self._explore_less(),
                                lambda s: plotter.add_to_curr_batch(datetime.now()),
                                lambda s: logger.remember(s["reward"])]  # Remember for future logging
        after_episode_callbacks = [lambda s: plotter.finish_curr_batch(),
                                   lambda s: logger.log(f"Score: {np.sum(logger.memory)} \t Episode: {s['e']}"),
                                   lambda s: logger.forget()]  # Empty the memory after taking the sum
        after_gameloop_callbacks = [lambda s: plotter.plot(aggregator=get_timeseq_diff)]
                                        # plot the lenghts of the games (differences of each sequence)

        self._play_through(gym_env, n_episodes=n_episodes, max_episode_length=max_episode_length, save_path=save_path,
                           load_path=load_path, after_step_callbacks=after_step_callbacks,
                           after_episode_callbacks=after_episode_callbacks,
                           after_gameloop_callbacks=after_gameloop_callbacks)

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
                           explore=False, after_step_callbacks=[lambda s: gym_env.render()])

import gym
from basic_dqn_agent import DQNAgent

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(8, 4)
    agent.train(env, n_episodes=100)
    #agent.play(env)

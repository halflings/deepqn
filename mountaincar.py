import gym
from agent import Config, Agent

import sys

env = gym.make('MountainCar-v0')
state_shape = env.observation_space.shape
assert len(state_shape) == 1
config = Config(state_dim=state_shape[0], n_actions=env.action_space.n, batch_size=64,
                discount=0.9, learning_rate=0.00001, epsilon_initial=0.99, epsilon_minimum=0.0,
                epsilon_decay_steps=2000, epsilon_decay_rate=0.95, memory_size=10000,
                training_period=500, layer_sizes=[50, 50, 50], debug_logging_period=4)
agent = Agent(config)
agent.train(env, max_episodes=80, max_steps=8000,
            reward_func=lambda r, s, t: 1000 if t else abs(s[1]),
            submit='--submit' in sys.argv)

import gym
from agent import Config, Agent

env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape
assert len(state_shape) == 1
config = Config(state_dim=state_shape[0], n_actions=env.action_space.n, batch_size=256,
                discount=0.8, learning_rate=0.001, epsilon_initial=0.99, epsilon_minimum=0.01,
                epsilon_decay_steps=700, epsilon_decay_rate=0.95, memory_size=5000,
                training_period=50, layer_sizes=[80, 200])
agent = Agent(config)
agent.train(env, max_steps=500, max_episodes=1400)

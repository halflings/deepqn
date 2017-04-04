import gym
from agent import Config, Agent

env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape
assert len(state_shape) == 1
config = Config(state_dim=state_shape[0], n_actions=env.action_space.n, batch_size=64,
                discount=0.99, learning_rate=0.00001, epsilon=0.05, memory_size=200,
                training_period=50)
agent = Agent(config)
agent.train(env, max_episodes=1000)

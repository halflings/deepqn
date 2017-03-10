import numpy as np


class Config:

    def __init__(self, state_dim, n_actions, memory_size, batch_size):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size


class ReplayMemory:

    def __init__(self, memory_size, state_dim, n_actions):
        self.memory_size = memory_size
        self.states = np.zeros([memory_size, state_dim])
        self.actions = np.zeros([memory_size, n_actions])
        self.rewards = np.zeros([memory_size, 1])
        self.terminals = np.zeros([memory_size, 1])
        self.cur_index = 0
        self.actual_size = 0

    def add(self, state, action, reward, terminal):
        self.states[self.cur_index] = state
        self.actions[self.cur_index] = action
        self.rewards[self.cur_index] = reward
        self.terminals[self.cur_index] = terminal
        self.cur_index = (self.cur_index + 1) % self.memory_size
        self.actual_size = min(self.actual_size + 1, self.memory_size)

    def sample(self, k):
        assert self.actual_size > k
        indices = np.random.randint(0, self.actual_size, size=k)
        return (self.states[indices - 1], self.actions[indices],
                self.states[indices], self.rewards[indices])


class Agent:

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.memory = ReplayMemory(config.memory_size, config.state_dim, config.n_actions)
        self.build_model()

    def build_model(self):
        self.pre_state = tf.placeholder([self.config.batch_size, self.config.dim_states])
        self.post_state = tf.placeholder([self.config.batch_size, self.config.dim_states])
        self.action_prob = tf.placeholder([self.config.batch_size, self.config.dim_actions])

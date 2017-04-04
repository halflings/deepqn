import numpy as np
import tensorflow as tf


class Config:

    def __init__(self, state_dim, n_actions, memory_size, batch_size, discount,
                 learning_rate, epsilon, training_period):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.training_period = training_period


class ReplayMemory:

    def __init__(self, memory_size, state_dim):
        self.memory_size = memory_size
        self.states = np.zeros([memory_size, state_dim])
        self.actions = np.zeros([memory_size])
        self.rewards = np.zeros([memory_size])
        self.states_prime = np.zeros([memory_size, state_dim])
        self.terminals = np.zeros([memory_size])
        self.cur_index = 0
        self.actual_size = 0

    def add(self, s_t, a_t, r_t, s_t_prime, terminal):
        self.states[self.cur_index] = s_t
        self.actions[self.cur_index] = a_t
        self.rewards[self.cur_index] = r_t
        self.states_prime[self.cur_index] = s_t_prime
        self.terminals[self.cur_index] = terminal
        self.cur_index = (self.cur_index + 1) % self.memory_size
        self.actual_size = min(self.actual_size + 1, self.memory_size)

    def sample(self, k):
        assert self.actual_size > k
        indices = np.random.randint(0, self.actual_size, size=k)
        # s_t, a_t, r_t, s_t_prime, terminal_t_prime
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.states_prime[indices], self.terminals[indices])


def huber_loss(delta):
    return tf.where(tf.abs(delta) < 1.0, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)


def linear_layer(inputs, units, activation, batch_norm=True):
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.1)
    if batch_norm:
        inputs = tf.layers.batch_normalization(inputs)
    out = tf.layers.dense(inputs=inputs, units=units,
                          activation=activation, kernel_initializer=w_init,
                          bias_initializer=b_init,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer,
                          bias_regularizer=tf.contrib.layers.l2_regularizer)
    return out


class Agent:

    def __init__(self, config):
        self.config = config
        self.memory = ReplayMemory(config.memory_size, config.state_dim)
        self.__build_model()
        self.session = None
        self.step = 0

    def __build_model(self):
        # Inputs
        self.state = tf.placeholder(
            tf.float32, shape=[None, self.config.state_dim])
        self.action = tf.placeholder(tf.int64, [None], name='action')
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')

        # Feed-forward net
        hidden1 = linear_layer(self.state, units=100, activation=tf.nn.relu)
        hidden2 = linear_layer(hidden1, units=256, activation=tf.nn.relu)
        self.q = linear_layer(hidden2, units=self.config.n_actions, activation=tf.nn.relu)
        self.q_max = tf.reduce_max(self.q)
        self.argmax_q = tf.argmax(self.q, axis=1)

        # Loss and training
        with tf.variable_scope('loss-training'):
            action_one_hot = tf.one_hot(
                self.action, self.config.n_actions, 1.0, 0.0, name='action_one_hot')
            predicted_q_a = tf.reduce_sum(
                self.q * action_one_hot, reduction_indices=1, name='predicted_q_a')

            self.loss = tf.reduce_mean(huber_loss(self.target_q - predicted_q_a), name='loss')
            self.global_step = tf.Variable(0, trainable=False)
            # Optimizer
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

    def __train_mini_batch(self):
        s_t, a_t, r_t, s_t_prime, terminal = self.memory.sample(self.config.batch_size)
        q_prime_max = self.session.run(self.q_max, {self.state: s_t_prime})
        target_q = r_t + (1. - terminal) * self.config.discount * q_prime_max
        _, q_t, loss = self.session.run([self.optim, self.q, self.loss], {
            self.target_q: target_q,
            self.action: a_t,
            self.state: s_t
        })
        sample_i = np.random.randint(0, len(q_t))
        sample_qt, sample_at = q_t[sample_i], a_t[sample_i]
        print("Loss = {:.2f} ; Reward = {:.2f}, sample: q_t = {}, a_t = {}".format(
            loss, np.mean(r_t), sample_qt, sample_at))

    def predict(self, state, epsilon=None):
        epsilon = epsilon or self.config.epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(self.config.n_actions)
        else:
            action = self.session.run(self.argmax_q, {self.state: [state]})[0]
        return action

    def observe(self, s_t, a_t, r_t, s_t_prime, terminal):
        self.memory.add(s_t, a_t, r_t, s_t_prime, terminal)
        self.step += 1
        if self.step % self.config.training_period == 0 and \
                self.memory.actual_size > self.config.batch_size:
            self.__train_mini_batch()

    def train(self, env, max_episodes=100, max_steps=300):
        self.step = 0
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer()])

        for i_episode in range(max_episodes):
            s_t, s_t_prime = env.reset(), None
            for t in range(max_steps):
                # Rendering only the last 10 episodes
                if max_episodes - i_episode <= 10:
                    env.render()
                epsilon = 1.0 if i_episode < 10 else self.config.epsilon
                a_t = self.predict(s_t, epsilon=epsilon)
                s_t_prime, r_t, terminal, info = env.step(a_t)
                self.observe(s_t, a_t, r_t, s_t_prime, terminal)
                s_t = s_t_prime
                if terminal:
                    if i_episode % 10 == 0:
                        print("{}# Finished at: {}".format(i_episode, t + 1))
                    break

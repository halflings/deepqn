import gym
from gym import wrappers
import numpy as np
import tensorflow as tf

import os
import string

from secrets import OPENAI_GYM_API_KEY


class Config:

    def __init__(self, state_dim, n_actions, memory_size, batch_size, discount,
                 learning_rate, epsilon_initial, epsilon_decay_steps, epsilon_decay_rate,
                 epsilon_minimum, training_period, layer_sizes, summary_root='/tmp/tf-summary',
                 summary_period=200, debug_logging_period=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_minimum = epsilon_minimum
        self.training_period = training_period
        self.layer_sizes = layer_sizes
        self.summary_root = summary_root
        self.summary_dir = self.__get_run_summary_dir(self.summary_root)
        self.summary_period = summary_period
        self.debug_logging_period = debug_logging_period

    @staticmethod
    def __get_run_summary_dir(summary_root):
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        num_dirs = [int(d) for d in os.listdir(summary_root)
                    if all(c in string.digits for c in d)]
        next_dir_num = max(num_dirs) + 1 if num_dirs else 0
        next_dir = os.path.join(summary_root, str(next_dir_num))
        os.makedirs(next_dir)
        return next_dir


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

    def __build_model(self):
        self.session = tf.Session()

        # Inputs
        self.state = tf.placeholder(
            tf.float32, shape=[None, self.config.state_dim], name='state')
        self.action = tf.placeholder(tf.int64, [None], name='action')
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')

        # Feed-forward net
        cur_input = self.state
        for layer_size in self.config.layer_sizes:
            cur_input = linear_layer(cur_input, units=layer_size, activation=tf.nn.relu)
        self.q = linear_layer(cur_input, units=self.config.n_actions, activation=tf.nn.relu)
        self.q_max = tf.reduce_max(self.q, axis=1)
        self.argmax_q = tf.argmax(self.q, axis=1)
        tf.summary.histogram('q_max', self.q_max)
        tf.summary.histogram('argmax_q', self.argmax_q)

        with tf.variable_scope('misc'):
            self.step = tf.Variable(1, name='step', trainable=False, dtype=tf.int32)
            self.increment_step_op = tf.assign(self.step, self.step + 1)
            self.epsilon = tf.maximum(self.config.epsilon_minimum,
                                      tf.train.exponential_decay(
                                          self.config.epsilon_initial,
                                          self.step,
                                          self.config.epsilon_decay_steps,
                                          self.config.epsilon_decay_rate))
            tf.summary.scalar('epsilon', self.epsilon)

        # Loss and training
        with tf.variable_scope('loss-training'):
            action_one_hot = tf.one_hot(
                self.action, self.config.n_actions, 1.0, 0.0, name='action_one_hot')
            predicted_q_a = tf.reduce_sum(
                self.q * action_one_hot, axis=1, name='predicted_q_a')
            tf.summary.histogram('q_a', predicted_q_a)

            self.loss = tf.reduce_mean(huber_loss(self.target_q - predicted_q_a), name='loss')
            tf.summary.scalar('loss', self.loss)
            # Optimizer
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

        # Summary
        self.writer = tf.summary.FileWriter(self.config.summary_dir, self.session.graph)
        self.summaries = tf.summary.merge_all()

    def __train_mini_batch(self):
        s_t, a_t, r_t, s_t_prime, terminal = self.memory.sample(self.config.batch_size)
        q_prime_max = self.session.run(self.q_max, {self.state: s_t_prime})
        target_q = r_t + (1. - terminal) * self.config.discount * q_prime_max
        feed_dict = {
            self.target_q: target_q,
            self.action: a_t,
            self.state: s_t
        }
        _, q_t, loss = self.session.run(
            [self.optim, self.q, self.loss], feed_dict=feed_dict)
        if self.step.eval(self.session) % self.config.summary_period == 0:
            summary = self.session.run(self.summaries, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.step.eval(self.session))

        # Debug logging
        if self.config.debug_logging_period and \
                self.session.run(self.step) % self.config.debug_logging_period == 0:
            sample_i = np.random.randint(0, len(q_t))
            sample_qt, sample_at = q_t[sample_i], a_t[sample_i]
            if self.session.run(self.step) % 4 == 0:
                print("Loss = {:.2f} ; Reward = {:.2f}, sample: q_t = {}, a_t = {}".format(
                    loss, np.mean(r_t), sample_qt, sample_at))

    def predict(self, state, epsilon=None):
        epsilon = epsilon or self.session.run(self.epsilon)
        if np.random.random() < epsilon:
            action = np.random.randint(self.config.n_actions)
        else:
            action = self.session.run(self.argmax_q, {self.state: [state]})[0]
        return action

    def observe(self, s_t, a_t, r_t, s_t_prime, terminal):
        self.memory.add(s_t, a_t, r_t, s_t_prime, terminal)
        self.session.run(self.increment_step_op)
        if self.step.eval(self.session) % self.config.training_period == 0 and \
                self.memory.actual_size > self.config.batch_size:
            self.__train_mini_batch()

    def train(self, env, max_episodes=100, max_steps=300, reward_func=None, submit=False):
        if submit:
            monitor_dir = '/tmp/env_monitor'
            env = wrappers.Monitor(env, monitor_dir, force=True)
        self.session.run([tf.global_variables_initializer()])
        episode_ends = []
        for i_episode in range(max_episodes):
            s_t, s_t_prime = env.reset(), None
            for t in range(max_steps):
                # Rendering only the last 10 episodes
                if max_episodes - i_episode <= 10:
                    env.render()
                a_t = self.predict(s_t)
                s_t_prime, r_t, terminal, info = env.step(a_t)
                r_t = reward_func(r_t, s_t, terminal) if reward_func else r_t
                self.observe(s_t, a_t, r_t, s_t_prime, terminal)
                s_t = s_t_prime
                if terminal:
                    break
            episode_ends.append(t + 1)
            if i_episode % 10 == 0:
                print("{}# Last 10 finished at: {}".format(i_episode, episode_ends[-10:]))

        if submit:
            env.close()
            if submit:
                gym.upload(monitor_dir, api_key=OPENAI_GYM_API_KEY)

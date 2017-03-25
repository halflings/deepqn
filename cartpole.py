import gym
from agent import Config, Agent


env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape
assert len(state_shape) == 1
config = Config(state_dim=state_shape[0], n_actions=env.action_space.n, batch_size=20, discount=0.9,
                learning_rate=0.01, epsilon=0.05, memory_size=100, training_period=10)
agent = Agent(config)
agent.initialize()

GLOBAL_I = 0
for i_episode in range(300):
    s_t = env.reset()
    for t in range(500):
        GLOBAL_I += 1
        # Skipping some frames
        if GLOBAL_I % 2 == 0:
            env.render()
        a_t = agent.predict(s_t)
        s_t, r_t, terminal, info = env.step(a_t)
        agent.observe(s_t, a_t, r_t, terminal)
        if terminal:
            print("{}# Finished at: {}".format(i_episode, t + 1))
            break

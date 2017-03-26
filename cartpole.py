import gym
from agent import Config, Agent


env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape
assert len(state_shape) == 1
config = Config(state_dim=state_shape[0], n_actions=env.action_space.n, batch_size=64,
                discount=0.8, learning_rate=0.01, epsilon=0.05, memory_size=200,
                training_period=20)
agent = Agent(config)
agent.initialize()

GLOBAL_I = 0
MAX_EPISODES = 800
for i_episode in range(MAX_EPISODES):
    s_t = env.reset()
    for t in range(500):
        GLOBAL_I += 1
        # Rendering only the last 10 episodes
        if MAX_EPISODES - i_episode <= 10:
            env.render()
        a_t = agent.predict(s_t, epsilon=1.0 if i_episode < 20 else None)
        s_t, r_t, terminal, info = env.step(a_t)
        agent.observe(s_t, a_t, r_t, terminal)
        if terminal:
            if i_episode % 10 == 0:
                print("{}# Finished at: {}".format(i_episode, t + 1))
            break

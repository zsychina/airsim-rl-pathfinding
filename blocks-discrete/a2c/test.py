import sys
sys.path.append('..')
from environment import Env
from agent import A2C

max_episode = 10

env = Env()
a2c = A2C(env=env)

a2c.load()
for episode_i in range(max_episode):
    print(f'episode {episode_i}...')
    ep_reward = a2c.test_env_episode()
    print(f'episode {episode_i} reward {ep_reward}')
import sys
sys.path.append('..')
from environment import Env
from agent import DQN

max_episode = 10
dqn = DQN()
env = Env()

dqn.load()

for episode_i in range(max_episode):
    print(f'episode {episode_i}')
    state = env.reset()
    ep_reward = 0
    while True:
        action = dqn.select_action_test(state)
        next_state, reward, done = env.step(action)
        ep_reward += reward
        if done:
            break
        state = next_state     


import sys
sys.path.append('..')
from environment import Env
from agent import DQN, MEMORY_CAPACITY
import logging

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')

max_episode = 1000

dqn = DQN()
env = Env()

for episode_i in range(max_episode):
    print(f'episode {episode_i}')
    state = env.reset()
    ep_reward = 0
    while True:
        action = dqn.select_action(state)
        next_state, reward, done = env.step(action)
        
        dqn.store_transition(state[0], state[1], action, reward, next_state[0], next_state[1])
        ep_reward += reward
        
        if dqn.memory_counter >= MEMORY_CAPACITY:
            dqn.learn()
            if done:
                dqn.save()

        if done:
            print(f'episode {episode_i}, reward {ep_reward}')
            logging.info(f'{episode_i}: {ep_reward}')
            break
        
        state = next_state        



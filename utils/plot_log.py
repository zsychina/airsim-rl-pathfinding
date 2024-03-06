import pandas as pd
import matplotlib.pyplot as plt

project = 'blocks-discrete'
algorithm = 'dqn'
path = f'../{project}/{algorithm}/log.txt'

ep_rewards = pd.read_csv(path, sep=':', header=None, names=['episode', 'reward'])

plt.figure(figsize=(10, 6))
plt.plot(ep_rewards['episode'], ep_rewards['reward'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes')
plt.grid(True)
plt.show()


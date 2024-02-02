from agent import A2C
import torch.optim as optim
import math
import sys
sys.path.append('..')
from environment import Env

LR = 1e-3
MAX_EPISODE = 10000

env = Env()
agent = A2C(env=env)

actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

r = []
avg_r = []
max_r = -math.inf

for i in range(MAX_EPISODE):
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    
    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode()
    
    r.append(rewards)
    
    if len(r) >= 100:
        episode_count = i - (i % 100)
        prev_episodes = r[len(r) - 100:]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(f'Average reward during episodes {episode_count}-{episode_count + 100} is {avg_r.item()}')
            
    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)
    
    l_actor.backward()
    l_critic.backward()
    
    actor_optim.step()
    critic_optim.step()
    

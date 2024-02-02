from agent import A2C
import torch.optim as optim
import torch
import math
import sys
sys.path.append('..')
from environment import Env

LR = 1e-3
MAX_EPISODE = 10000

env = Env()
agent = A2C(env=env)

agent.actor.load_state_dict(torch.load('checkpoint/actor.pth'))
agent.critic.load_state_dict(torch.load('checkpoint/critic.pth'))

actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

r = []
avg_r = []
max_r = -math.inf

for i in range(MAX_EPISODE):
    print(f'episode {i}')
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    
    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode()
    
    r.append(rewards)
    
    avg_reward = total_reward / len(rewards)
    print(f'Average reward for episode {i} is {avg_reward}')
    
    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)
    
    l_actor.backward()
    l_critic.backward()
    
    actor_optim.step()
    critic_optim.step()
    
    torch.save(agent.actor.state_dict(), 'checkpoint/actor.pth')
    torch.save(agent.critic.state_dict(), 'checkpoint/critic.pth')
    
    

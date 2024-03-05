import sys
sys.path.append('..')
from environment import Env
from agent import PPO
import torch
import logging

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')
torch.manual_seed(42)

MAX_EPISODE = 10000
UPDATE_TIMESTEP = 2000
MODEL_SAVE_TIMESTEP = 5000

env = Env()

K_epochs = 40
eps_clip = 0.2
gamma = .99
lr_actor =  3e-4
lr_cirtic = 1e-3
action_dim = 8

ppo_agent = PPO(
    action_dim=action_dim, 
    lr_actor=lr_actor,
    lr_critic=lr_cirtic,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    gamma=gamma
)


time_step = 0
for episode_i in range(MAX_EPISODE):
    state = env.reset()
    ep_reward = 0
    
    done = False
    while not done:
        action = ppo_agent.select_action(state)
        state, reward, done = env.step(action)
        
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step += 1
        ep_reward += reward
        
        if time_step % UPDATE_TIMESTEP == 0:
            ppo_agent.update()

        if time_step % MODEL_SAVE_TIMESTEP == 0:
            ppo_agent.save()
            
    print(f'episdoe {episode_i} reward {ep_reward}')  
    logging.info(f'{episode_i}: {ep_reward}')  



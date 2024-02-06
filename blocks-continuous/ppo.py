from environment import Env
from agent import PPO
import torch


MAX_EPISODE = 10000
UPDATE_TIMESTEP = 2000
MODEL_SAVE_TIMESTEP = 5000
PRINT_TIMESTEP = 500

env = Env()

K_epochs = 40
eps_clip = 0.2
gamma = .99
lr_actor =  3e-4
lr_cirtic = 1e-3
state_dim = 10
action_dim = 4

ppo_agent = PPO(
    state_dim=state_dim, 
    action_dim=action_dim, 
    lr_actor=lr_actor,
    lr_critic=lr_cirtic,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    gamma=gamma
)


time_step = 0
running_reward = 0
running_episode = 0
for episode_i in range(MAX_EPISODE):
    state = env.reset()
    current_episode_reward = 0
    
    done = False
    while not done:
        action = ppo_agent.select_action(state)
        state, reward, done = env.step(action)
        
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step += 1
        current_episode_reward += reward
        
        if time_step % PRINT_TIMESTEP == 0:
            avg_reward = running_reward / running_episode
            print(f'Episode {episode_i}, average reward: {avg_reward}')
        
        if time_step % UPDATE_TIMESTEP == 0:
            ppo_agent.update()

        if time_step % MODEL_SAVE_TIMESTEP == 0:
            ppo_agent.save()
            
    running_reward += current_episode_reward
    running_episode += 1
    



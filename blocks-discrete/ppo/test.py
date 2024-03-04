import sys
sys.path.append('..')
from environment import Env
from agent import PPO
import torch

torch.manual_seed(42)

MAX_EPISODE = 10
PRINT_TIMESTEP = 500

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

ppo_agent.load()

for episode_i in range(MAX_EPISODE):
    state = env.reset()
    ep_reward = 0
    done = False
    while not done:
        action = ppo_agent.select_action(state)
        state, reward, done = env.step(action)
        ep_reward += reward
        
    print(f'episode {episode_i} reward {ep_reward}')
        

    



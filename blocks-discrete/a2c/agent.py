import torch
import torch.nn as nn

from torch.distributions import Categorical

from model import Actor, Critic

device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_dim = 4

class A2C(nn.Module):
    
    def __init__(self, env, gamma=0.99):
        super().__init__()
        
        self.env = env
        self.gamma = gamma
        
        self.actor = Actor(action_dim).to(device)
        self.critic = Critic().to(device)

    def train_env_episode(self):
        rewards = []
        critic_vals = []
        action_lp_vals = []
        
        state = self.env.reset()
        done = False
        
        while not done:
            image = torch.from_numpy(state[0]).float().to(device)
            location = torch.from_numpy(state[1]).float().to(device)
            
            action_logits = self.actor(image, location)
            
            action = Categorical(logits=action_logits).sample()
            
            action_log_prob = action_logits[action]
            pred = torch.squeeze(self.critic(image, location).view(-1))
            
            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)
            
            state, reward, done = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())
            
        total_reward = sum(rewards)
        
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G
            
        def f(inp):
            return torch.stack(tuple(inp), 0)
        
        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards + 1e-6))
        
        return rewards, f(critic_vals), f(action_lp_vals), total_reward

    @staticmethod
    def compute_loss(action_p_vals, G, V, crititc_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach()
        return -(torch.sum(action_p_vals * advantage)), crititc_loss(G, V)
            
        
        
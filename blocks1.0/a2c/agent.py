import torch
import torch.nn as nn

from torch.distributions import Categorical

n_observation = 10
n_action = 4

class A2C(nn.Module):
    
    def __init__(self, env, gamma=0.99):
        super().__init__()
        
        self.env = env
        self.gamma = gamma
        self.n_input = n_observation
        self.n_output = n_action
        
        self.actor = nn.Sequential(
            nn.Linear(self.n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_output)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).double()
        
    def train_env_episode(self):
        rewards = []
        critic_vals = []
        action_lp_vals = []
        
        observation = self.env.reset()
        done = False
        
        while not done:
            observation = torch.from_numpy(observation).double()
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()
            
            action_log_prob = action_logits[action]
            pred = torch.squeeze(self.critic(observation).view(-1))
            
            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)
            
            observation, reward, done = self.env.step(action.item())
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
            
        
        
        
        
        




import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = nn.Tanh(self.linear1(x))
        x = nn.Tanh(self.linear2(x))
        x = nn.Tanh(self.linear3(x))
        x = nn.Tanh(self.linear4(x))
        x = nn.Tanh(self.linear5(x))
        return x
        
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = nn.Tanh(self.linear1(x))
        x = nn.Tanh(self.linear2(x))
        x = nn.Tanh(self.linear3(x))
        x = nn.Tanh(self.linear4(x))
        x = self.linear5(x)
        return x
        




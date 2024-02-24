# continuous PPO
import torch
import torch.nn as nn
from torch.distributions import Categorical
# from torch.distributions import MultivariateNormal
from model import Actor, Critic


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.imgs = []
        self.locs = []
        
        
    def clear(self):
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.imgs[:]
        del self.locs[:]


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.actor = Actor(action_dim)
        self.critic = Critic()
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, image, location):
        action_probs = self.actor(image, location)
        dist = Categorical(logits=action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(image, location)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, img, loc, action):
        action_probs = self.actor(img, loc)
        dist = Categorical(logits=action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(img, loc)
                
        return action_logprobs, state_values, dist_entropy
    
    
class PPO:
    def __init__(self, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            image = torch.from_numpy(state[0]).float().to(device)
            location = torch.from_numpy(state[1]).float().to(device)
            action, action_logprob, state_val = self.policy_old.act(image, location)
             
        self.buffer.imgs.append(image)
        self.buffer.locs.append(location)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_imgs = torch.squeeze(torch.stack(self.buffer.imgs, dim=0)).detach().to(device)
        old_locs = torch.squeeze(torch.stack(self.buffer.locs, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_imgs, old_locs, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
    def save(self):
        torch.save(self.policy_old.state_dict(), './checkpoint/policy.pth')
   

    def load(self):
        self.policy_old.load_state_dict(torch.load('./checkpoint/policy.pth', map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load('./checkpoint/policy.pth', map_location=lambda storage, loc: storage))
        
        
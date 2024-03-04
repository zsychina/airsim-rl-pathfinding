from model import Net
import numpy as np
import torch

BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 20000
Q_NETWORK_ITERATION = 100

action_dim = 8


class DQN:
    def __init__(self):
        self.eval_net = Net(action_dim)
        self.target_net = Net(action_dim)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.image_buffer = np.zeros((MEMORY_CAPACITY, 3, 64, 64))
        self.location_buffer = np.zeros((MEMORY_CAPACITY, 12))
        self.action_buffer = np.zeros((MEMORY_CAPACITY, 1))
        self.reward_buffer = np.zeros((MEMORY_CAPACITY, 1))
        self.next_image_buffer = np.zeros((MEMORY_CAPACITY, 3, 64, 64))
        self.next_location_buffer = np.zeros((MEMORY_CAPACITY, 12))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()

    def select_action(self, state):
        image = torch.from_numpy(state[0]).float()
        location = torch.from_numpy(state[1]).float()
        if np.random.randn() <= EPISILO:
            action_values = self.eval_net(image, location)
            action = action_values.argmax()
        else:
            action = np.random.randint(0, action_dim)
        return action

    def store_transition(self, image, location, action, reward, next_image, next_location):
        index = self.memory_counter % MEMORY_CAPACITY
        self.image_buffer[index] = image
        self.location_buffer[index] = location
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_image_buffer[index] = next_image
        self.next_location_buffer[index] = next_location
        
        self.memory_counter += 1
        
    def learn(self):
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.eval_net.load_state_dict(self.target_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        image_batch = torch.FloatTensor(self.image_buffer[sample_index])
        location_batch = torch.FloatTensor(self.location_buffer[sample_index])
        action_batch = torch.LongTensor(self.action_buffer[sample_index])
        reward_batch = torch.FloatTensor(self.reward_buffer[sample_index])
        next_image_batch = torch.FloatTensor(self.next_image_buffer[sample_index])
        next_location_batch = torch.FloatTensor(self.next_location_buffer[sample_index])
        
        q_eval = self.eval_net(image_batch, location_batch).gather(1, action_batch)
        q_next = self.target_net(next_image_batch, next_location_batch).detach()
        q_target = reward_batch + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_fn(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        torch.save(self.eval_net.state_dict(), 'checkpoint/eval.pth')
        torch.save(self.target_net.state_dict(), 'checkpoint/target.pth')
        
    def load(self):
        self.eval_net.load_state_dict(torch.load('checkpoint/eval.pth'))
        self.target_net.load_state_dict(torch.load('checkpoint/target.pth'))
            

import logging
import airsim
from environmet import Env
import sys
import random
import math
from helper import ReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append('..')
from models.fc import FC

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('blocks')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using {device}')

env = Env()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# temp state_n := 10
# temp action_n := 6
state_n = 10
action_n = 10

state = env.reset()

policy_net = FC(state_n, action_n).to(device)
target_net = FC(state_n, action_n).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * env.step_count / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # return policy_net(state).max(1).indices.view(1, 1)
            pass
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        pass
    
def optimizer():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    
    
    

import airsim
import torch
import logging
import sys
sys.path.append('..')
from models.fc import FC
from environmet import Env
from itertools import count

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('blocks')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using {device}')

client = airsim.MultirotorClient()

state_n = 10
action_n = 4

policy_net = FC(state_n, action_n).to(device)
# policy_net.load_state_dict(torch.load('policy_net.pth'))
policy_net = torch.load('policy_net.pth')

def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

env = Env()

for i in range(100):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)

        observation, reward, terminated = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated

        drone_state = client.getMultirotorState()
        logger.warn(f'reward {reward.item()}')
        if drone_state.collision.has_collided:
            logger.warn(drone_state)

        if done:    
            break

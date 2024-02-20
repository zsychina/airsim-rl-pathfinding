from environment import Env
from agent import PPO

MAX_EPISODE_LEN = 400
TEST_EPISODE = 10

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

ppo_agent.load()

test_running_rewards = 0

for episode_i in range(1, TEST_EPISODE + 1):
    episode_reward = 0
    state = env.reset()
    
    for t in range(1, MAX_EPISODE_LEN):
        action = ppo_agent.select_action(state)
        state, reward, done = env.step(action)
        
        episode_reward += reward
        
        if done:
            break
        
    ppo_agent.buffer.clear()
    test_running_rewards += episode_reward
    print(f'Episode {episode_i}, Reward {episode_reward}')



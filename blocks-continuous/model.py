import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # image
        # 3x64x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=10, stride=2) # 4x28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6, stride=1)  # 8x23x23
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1) # 16x21x21
        self.linear_dv1 = nn.Linear(7056, 800)
        self.linear_dv2 = nn.Linear(800, 64)
        self.linear_dv3 = nn.Linear(64, 64)
        
        # location
        self.linear_l1 = nn.Linear(6, 16)
        self.linear_l2 = nn.Linear(16, 16)
        self.linear_l3 = nn.Linear(16, 8)
        
        # shared network
        self.linear_sn1 = nn.Linear(72, 64)
        self.linear_sn2 = nn.Linear(64, 32)
        self.linear_sn3 = nn.Linear(32, action_dim)
        
    def forward(self, x):
        depth_vision = x[0]
        locations = x[1]
        
        # process depth vision
        depth_vision = F.relu(self.conv1(depth_vision))
        depth_vision = F.relu(self.conv2(depth_vision))
        depth_vision = F.relu(self.conv3(depth_vision))
        depth_vision = depth_vision.view(-1, 7056)
        depth_vision = F.relu(self.linear_dv1(depth_vision))
        depth_vision = F.relu(self.linear_dv2(depth_vision))
        depth_vision = F.relu(self.linear_dv3(depth_vision))
        
        # process location
        locations = F.relu(self.linear_l1(locations))
        locations = F.relu(self.linear_l2(locations))
        locations = F.relu(self.linear_l3(locations))
        locations = locations.unsqueeze(0)
        
        # process shared network
        feature = torch.cat([depth_vision, locations], dim=1)
        feature = F.relu(self.linear_sn1(feature))
        feature = F.relu(self.linear_sn2(feature))
        feature = torch.tanh(self.linear_sn3(feature))
        return feature
        
        
        
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # image
        # 3x64x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=10, stride=2) # 4x28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6, stride=1)  # 8x23x23
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1) # 16x21x21
        self.linear_dv1 = nn.Linear(7056, 800)
        self.linear_dv2 = nn.Linear(800, 64)
        self.linear_dv3 = nn.Linear(64, 64)
        
        # location
        self.linear_l1 = nn.Linear(6, 16)
        self.linear_l2 = nn.Linear(16, 16)
        self.linear_l3 = nn.Linear(16, 8)
        
        # shared network
        self.linear_sn1 = nn.Linear(72, 64)
        self.linear_sn2 = nn.Linear(64, 32)
        self.linear_sn3 = nn.Linear(32, 1)
        
    def forward(self, x):
        depth_vision = x[0]
        locations = x[1]
        
        # process depth vision
        depth_vision = F.relu(self.conv1(depth_vision))
        depth_vision = F.relu(self.conv2(depth_vision))
        depth_vision = F.relu(self.conv3(depth_vision))
        depth_vision = depth_vision.view(-1, 7056)
        depth_vision = F.relu(self.linear_dv1(depth_vision))
        depth_vision = F.relu(self.linear_dv2(depth_vision))
        depth_vision = F.relu(self.linear_dv3(depth_vision))
        
        # process location
        locations = F.relu(self.linear_l1(locations))
        locations = F.relu(self.linear_l2(locations))
        locations = F.relu(self.linear_l3(locations))
        locations = locations.unsqueeze(0)
        
        # process shared network
        feature = torch.cat([depth_vision, locations], dim=1)
        feature = F.relu(self.linear_sn1(feature))
        feature = F.relu(self.linear_sn2(feature))
        feature = self.linear_sn3(feature)
        return feature
    
    
if __name__ == '__main__':
    import numpy as np
    import cv2
    img = np.random.randint(0, 255, (3, 144, 256))
    loc = np.random.randn(6)
    img_transposed = img.transpose(1, 2, 0).astype(np.uint8)
    resized_image = cv2.resize(img_transposed, (64, 64))
    resized_image = resized_image.transpose(2, 0, 1)
    observation = [torch.from_numpy(resized_image).float().to('cpu'), torch.from_numpy(loc).float().to('cpu')]
    print(observation)
    actor = Actor(4)
    crtic = Critic()
    action = actor(observation)
    value = crtic(observation)
    print(action.detach().cpu().numpy().flatten())
    print(value.detach().cpu().numpy().flatten())
    
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
        
    def forward(self, images, locations):

        # process images
        images = F.relu(self.conv1(images))
        images = F.relu(self.conv2(images))
        images = F.relu(self.conv3(images))
        images = images.view(-1, 7056)
        images = F.relu(self.linear_dv1(images))
        images = F.relu(self.linear_dv2(images))
        images = F.relu(self.linear_dv3(images))
        images = images.squeeze()
        
        # process location
        locations = F.relu(self.linear_l1(locations))
        locations = F.relu(self.linear_l2(locations))
        locations = F.relu(self.linear_l3(locations))
        locations = locations.squeeze()
        
        # process shared network
        # print(images.shape, locations.shape)
        feature = torch.cat([images, locations], dim=-1)
        feature = F.relu(self.linear_sn1(feature))
        feature = F.relu(self.linear_sn2(feature))
        feature = self.linear_sn3(feature)
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
        
    def forward(self, images, locations):
        
        # process images
        images = F.relu(self.conv1(images))
        images = F.relu(self.conv2(images))
        images = F.relu(self.conv3(images))
        images = images.view(-1, 7056)
        images = F.relu(self.linear_dv1(images))
        images = F.relu(self.linear_dv2(images))
        images = F.relu(self.linear_dv3(images))
        images = images.squeeze()
        
        # process location
        locations = F.relu(self.linear_l1(locations))
        locations = F.relu(self.linear_l2(locations))
        locations = F.relu(self.linear_l3(locations))
        locations = locations.squeeze()
        
        # process shared network
        feature = torch.cat([images, locations], dim=-1)
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
    resized_image = torch.tensor(resized_image, dtype=torch.float32)
    loc = torch.tensor(loc, dtype=torch.float32)
    print(loc)
    
    actor = Actor(4)
    critic = Critic()
    action = actor(resized_image, loc)
    value = critic(resized_image, loc)
    print(action.detach().cpu().numpy().flatten())
    print(value.detach().cpu().numpy().flatten())
    
    
    img1 = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    loc1 = np.random.randn(6)
    loc2 = np.random.randn(6)
    
    imgs = [torch.from_numpy(img1).float(), torch.from_numpy(img2).float(), torch.from_numpy(img2).float()]
    locs = [torch.from_numpy(loc1).float(), torch.from_numpy(loc2).float(), torch.from_numpy(loc2).float()]
    old_imgs = torch.stack(imgs, dim=0)
    old_locs = torch.stack(locs, dim=0)
    print(old_locs)
    print(actor(old_imgs, old_locs))
    print(critic(old_imgs, old_locs))
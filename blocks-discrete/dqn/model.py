import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
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
        images = images.reshape(-1, 7056)
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
        
if __name__ == '__main__':
    import numpy as np
    import cv2
    import torch

    # env
    img = np.random.randint(0, 255, (3, 144, 256))
    loc = np.random.randn(3)
    img_transposed = img.transpose(1, 2, 0).astype(np.uint8)
    resized_image = cv2.resize(img_transposed, (64, 64))
    resized_image = resized_image.transpose(2, 0, 1)
    state = [resized_image, np.concatenate([loc, loc])]
    # print(state[0].shape, state[1].shape)

    # select action
    image = torch.from_numpy(state[0]).float()
    location = torch.from_numpy(state[1]).float()
    net = Net(8)
    action_values = net(image, location)
    action = action_values.argmax()
    # print(action)
    
    # learn
    image_batch = torch.FloatTensor(np.stack([state[0], state[0], state[0]]))
    location_batch = torch.FloatTensor(np.stack([state[1], state[1], state[1]]))
    action_batch = torch.LongTensor(np.stack([[3], [3], [3]]))
    print(image_batch.shape, location_batch.shape, action_batch.shape)
    q_evals = net(image_batch, location_batch).gather(1, action_batch)
    q_next = net(image_batch, location_batch)
    print(q_next)
    print(q_next.max(1)[0])
    
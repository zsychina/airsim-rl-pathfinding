import torch
import torch.nn as nn
import torch.nn.functional as F

# input 10
# output 6 
class FC(nn.Module):
    def __init__(self, input_n, output_n):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
        
if __name__ == '__main__':
    net = FC(4, 6)
    input = torch.tensor([1.0, 2.0, 3.0, 4.0])
    output = net(input)
    print(output)
        

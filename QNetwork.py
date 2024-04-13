
import torch.nn as nn
import torch.nn.functional as F
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_space=8):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, action_space)

    def forward(self, x):
        #flattening the tensor to 2D tensor batch_size*flattened_grid_size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)

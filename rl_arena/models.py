import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.ln1 = nn.Linear(state_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        return self.ln3(x)


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_actions):
        self.in_channels = in_channels
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)  # Add batch dim

        if x.size(1) != self.in_channels:
            x = x.permute(0, 3, 1, 2)  # B, C, H, W

        # Add prints for debugging
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

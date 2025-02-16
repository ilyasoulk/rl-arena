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
    def __init__(self, observation_space, hidden_dim, action_space):
        super().__init__()
        in_channels = observation_space[-1]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),  # Output: 23x23x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # Output: 10x10x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),  # Output: 8x8x32
            nn.ReLU(),
            nn.Flatten(),  # Output: 2048 (8*8*32)
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(hidden_dim, action_space)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)

        if x.max() > 1:
            x = x / 255.0

        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        features = self.features
        actions = self.head(features)
        return actions

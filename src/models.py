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
    def __init__(self, in_channels, hidden_dim, action_space):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # Output: 20x20x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: 9x9x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: 7x7x64
            nn.ReLU(),
            nn.Flatten(),  # Output: 3136 (7*7*64)
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(
            hidden_dim, action_space
        )  # Output: action_space (Q-values)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)  # Add batch dim

        if x.max() > 1:
            x = x / 255.0

        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        features = self.features(x)
        actions = self.head(features)
        return actions

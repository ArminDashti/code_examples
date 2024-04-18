import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_actions):
        super(QNetwork, self).__init__()
        self.linear = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear(x))
        x = self.linear(x)
        return x

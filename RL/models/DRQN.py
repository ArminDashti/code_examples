# This code was generated by Perplexity AI


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple

BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
UPDATE_EVERY = 4

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DRQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DRQN, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden(x.size(0))
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.fc(x[:, -1, :])
        return x, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

class Agent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.qnetwork_local = DRQN(state_size, action_size).to(device)
        self.qnetwork_target = DRQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayMemory(10000)
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        self.timestep = (self.timestep + 1) % UPDATE_EVERY
        if self.timestep == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            _, next_hidden = self.qnetwork_target(next_states)
        q_targets_next, _ = self.qnetwork_target(next_states, next_hidden)
        q_targets_next = q_targets_next.max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected, _ = self.qnetwork_local(states)
        q_expected = q_expected.gather(1, actions.long())
        loss = nn.MSELoss()(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
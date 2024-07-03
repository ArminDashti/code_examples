# This code was generated by OpenAI chatGPT 4o

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, atom_size, support):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.support = support
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim * atom_size)
        )
    
    def forward(self, x):
        q_atoms = self.fc(x).view(-1, self.output_dim, self.atom_size)
        dist = torch.softmax(q_atoms, dim=-1)
        q_values = torch.sum(dist * self.support, dim=-1)
        return dist, q_values


class C51Agent:
    def __init__(self, input_dim, output_dim, atom_size, v_min, v_max, gamma, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.lr = lr
        
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        self.delta_z = (v_max - v_min) / (self.atom_size - 1)
        
        self.online_net = Network(input_dim, output_dim, atom_size, self.support)
        self.target_net = Network(input_dim, output_dim, atom_size, self.support)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
    def update(self, state, action, reward, next_state, done):
        # Convert numpy arrays to torch tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        
        # Compute the target distribution
        with torch.no_grad():
            next_dist, _ = self.target_net(next_state)
            next_action = next_dist * self.support
            next_action = next_action.sum(2).max(1)[1]
            next_dist = next_dist[0, next_action]
            
            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            offset = torch.linspace(0, (self.atom_size - 1) * self.output_dim, self.atom_size).long().unsqueeze(0).expand(self.atom_size, -1)
            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
        dist, _ = self.online_net(state)
        log_p = torch.log(dist[0, action])
        elementwise_loss = -(proj_dist * log_p).sum(1)
        
        loss = elementwise_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sync_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, q_values = self.online_net(state)
        action = q_values.argmax().item()
        return action


def train_c51_agent(env, agent, episodes, sync_interval):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if episode % sync_interval == 0:
            agent.sync_target_net()
        
        print(f'Episode: {episode}, Total Reward: {total_reward}')
        
# Hyperparameters
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
atom_size = 51
v_min = -10
v_max = 10
gamma = 0.99
lr = 0.001
episodes = 500
sync_interval = 10

agent = C51Agent(input_dim, output_dim, atom_size, v_min, v_max, gamma, lr)
train_c51_agent(env, agent, episodes, sync_interval)


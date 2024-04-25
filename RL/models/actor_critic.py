# https://chat.openai.com/c/33a9ada9-8b93-48c6-8503-7e27c59144e0
# This code was generated from chatGPT 3.5

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor_critic = ActorCritic(input_dim, hidden_dim, output_dim)
        self.optimizer_actor = optim.Adam(self.actor_critic.fc_actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.fc_critic.parameters(), lr=lr_critic)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.actor_critic(state)
        action = torch.multinomial(action_probs, 1)
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action_probs, state_value = self.actor_critic(state)
        _, next_state_value = self.actor_critic(next_state)
        
        action = torch.LongTensor([[action]])
        action_probs = action_probs.gather(1, action)
        next_action_probs = self.actor_critic(next_state)[0].detach().max(1)[0]
        
        if done:
            target_value = torch.FloatTensor([[reward]])
        else:
            target_value = reward + self.gamma * next_state_value
        
        advantage = target_value - state_value
        actor_loss = -torch.log(action_probs) * advantage.detach()
        critic_loss = F.smooth_l1_loss(state_value, target_value.detach())
        
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()


state_dim = 4
action_dim = 2
hidden_dim = 128
lr_actor = 0.001
lr_critic = 0.001
gamma = 0.99

agent = ActorCriticAgent(state_dim, hidden_dim, action_dim, lr_actor, lr_critic, gamma)

num_episodes = 1000
for i_episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    print(f"Episode {i_episode+1}, Reward: {episode_reward}")

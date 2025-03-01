import random

import numpy as np
import torch
from agent_models import DQN
from base_agent import BaseAgent
import torch.optim as optim
import torch.nn as nn


class DQNAgent(BaseAgent):
    def __init__(self,  state_size, action_size, epsilon_start, epsilon_end, epsilon_decay, hidden_size, lr, gamma, device):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = device

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Network and target network
        self.model = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_model = DQN(state_size, action_size,
                                hidden_size).to(self.device)
        self.update_target_model()  # initialize target network

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Transform to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values for the taken actions
        q_values = self.model(states).gather(1, actions)
        # Next Q values from target network (detach to avoid gradient flow)
        next_q_values = self.target_model(
            next_states).detach().max(dim=1, keepdim=True)[0]
        # Compute target Q values using the Bellman equation
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

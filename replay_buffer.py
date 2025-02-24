import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size=100):
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample_batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)
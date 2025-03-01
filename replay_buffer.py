import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_batch = random.sample(
            self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*sample_batch)

        # Using np.stack for states and actions since they are numpy arrays with consistent shapes
        return (
            np.stack(states),              # shape: (batch_size, state_dim)
            np.stack(actions),             # shape: (batch_size, action_dim)
            np.array(rewards, dtype=np.float32),   # shape: (batch_size,)
            np.stack(next_states),         # shape: (batch_size, state_dim)
            np.array(dones, dtype=np.bool_)  # shape: (batch_size,)
        )

    def size(self):
        return len(self.buffer)

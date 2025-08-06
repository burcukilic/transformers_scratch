import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000, max_seq_len=100):
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.buffer = []
        self.position = 0

    def add_episode(self, states, actions, rewards, rtgs, tasks):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'returns_to_go': rtgs,
            'tasks': tasks
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
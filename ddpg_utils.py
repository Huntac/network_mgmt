import random
from collections import namedtuple, deque
import copy
import numpy as np
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("experience", ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)

    def sample(self, batch_size = 64):
        return random.sample(self.memory, k = self.batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma - sigma
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu-x) + self.sigma * np.random.randn(len(x))
        self.state = x+dx
        return self.state
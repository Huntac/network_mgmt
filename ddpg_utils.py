import random
from collections import namedtuple, deque
import copy
import numpy as np
class ReplayBuffer:


    """
    Buffer to store experiences for DDPG actor and critic training.
    """
    def __init__(self, buffer_size: int, batch_size: int):
        """
        Initialize an instance of ReplayBuffer.

            Args:
                buffer_size (int): number of experiences to store
                batch_size (int): number of experiences to return when sampled

            Usage Example:
                agent.replay_buffer = ReplayBuffer(100000, 128)
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("experience", ['state', 'action', 'reward', 'next_state', 'done'])


    def add(
        self, 
        state: np.ndarray,
        action: np.ndarray, 
        reward: np.ndarray, 
        next_state: np.ndarray, 
        done: bool):
        """
            Stores experiences as a named tuple in a deque.
                
            Each experience is added to the RHS. If self.memory 
            reaches buffer_size (it's max length) and a new 
            experience is appended, the oldest experience
            is dropped from the LHS.

                Args:
                    state (array-like): representation of the state 
                           before the action contained in this experience
                    action (array-like): representation of the agent's 
                           action in response to state
                    reward (array-like): representation of the reward 
                            produced by the agent's action 
                    next_state (array_like): representation of the 
                                state as a result of the agent's action
                    done bool: Usually boolean, was this 
                          experience the last experience in an episode?
        """
        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)


    def sample(self):
        """
        Return a sample from the buffer for training
        """
        return random.sample(self.memory, k = self.batch_size)


    def __len__(self):
        return len(self.memory)


class npReplayBuffer:
    """
    Implementation of replay buffer using numpy
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_center = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_center % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_center += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_center, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace = False)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
        

class OUNoise:


    """
    Instantiation of Ornstein - Uhlenbeck process 
    """
    def __init__(self, size: int, mu: float, theta: float, sigma: float):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # sets self.state such that self.mu remeains stationary throughout sampling
        self.reset()
    

    def reset(self):
        """
        Reset process to initial state
        """
        self.state = copy.copy(self.mu)


    def sample(self):
        """
        Step process, return new sample of OU noise process
        """
        x = self.state
        dx = self.theta * (self.mu-x) + self.sigma * np.random.randn(len(x))
        self.state = x+dx
        return self.state
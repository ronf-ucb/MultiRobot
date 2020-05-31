import random
from collections import namedtuple
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

# Store: true state, local states, local actions, local policies, goals, goal log_prob, goal means, reward, done, 
# next actions, next true states, next local states
Manager = namedtuple('Transition',
                        ('state', 'action', 'reward','next_state', 'mask'))
Worker = namedtuple('Transition', ('state', 'policy', 'goal', 'action', 'reward', 'mask', 'next_action', 'next_state'))

class WorkerMemory(object):
    def __init__(self):
        self.memory = []
        self.transition = Worker

    def push(self, s, policy, g, a, r, mask, next_action, next_state):
        """Saves a transition."""
        self.memory.append(self.transition(s, policy, g, a, r, mask, next_action, next_state))

    def sample(self, batch = 0):
        # Sample contiguous
        if batch == 0:
            transitions = self.transition(*zip(*self.memory))
            return transitions

        # Sample randomly
        c = np.random.choice(len(self.memory), batch)
        mem = map(self.memory.__getitem__, c)
        transitions = self.transition(*zip(*mem))
        return transitions

    def __len__(self):
        return len(self.memory)

class ManagerMemory(WorkerMemory):
    def __init__(self):
        self.memory = []
        self.transition = Manager

    def push(self, s, a, r, next_state, mask):
        self.memory.append(self.transition(s, a, r, next_state, mask))
    
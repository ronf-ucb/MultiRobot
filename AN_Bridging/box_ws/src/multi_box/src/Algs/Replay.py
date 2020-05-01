import numpy as np 
from collections import namedtuple, OrderedDict

labels = ('s', 'a', 'r', 'n_s', 'n_a', 'mask')

class Replay:
    def __init__(self, max_size):
        self.memory = []
        self.max = max_size
        self.curr = 0
    
    def push(self, experience):
        self.memory.append(experience)
        self.curr += 1
        if self.curr > self.max:
            del self.memory[0]

    def get_data(self, batch = -1):
        exp = zip(*self.memory)
        exp = OrderedDict(zip(labels, exp))
        state_batch = np.asarray(exp['s']).reshape(len(exp['s']), -1)
        action_batch = np.asarray(exp['a']).reshape(len(exp['a']), -1)
        reward_batch = np.asarray(exp['r']).reshape(len(exp['r']), -1)
        next_state_batch = np.asarray(exp['n_s']).reshape(len(exp['n_s']), -1)
        next_action_batch = np.asarray(exp['n_a']).reshape(len(exp['n_a']), -1)
        done_batch = np.asarray(exp['mask']).reshape(len(exp['mask']), -1)

        if batch != -1:
            c = np.random.choice(min(self.curr, self.max), batch)
            state_batch = state_batch[c]
            action_batch = action_batch[c]
            reward_batch = reward_batch[c]
            next_state_batch = next_state_batch[c]
            next_action_batch = next_action_batch[c]
            done_batch = done_batch[c]

        return state_batch, action_batch, reward_batch, next_state_batch, next_action_batch, done_batch

    def __len__(self):
        return len(self.buffer)
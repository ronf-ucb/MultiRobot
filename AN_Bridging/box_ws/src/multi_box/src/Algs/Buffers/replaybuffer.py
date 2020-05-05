import numpy as np


class ReplayBuffer(object):
    def __init__(self, size, a_dim, a_dtype, s_dim, s_dtype, store_mu =False):
        self.size     = size
        self.count    = 0  # the number of stored experiences
        self.head     = 0  # an index to which next experience is stored
        self.store_mu = store_mu
        
        self.s_t    = np.zeros(shape=(size, s_dim), dtype=s_dtype)
        self.a_t    = np.zeros(shape=(size, a_dim), dtype=a_dtype)
        self.r_t    = np.zeros(shape=size, dtype=np.float64)

        self.s_tp1  = np.zeros(shape=(size, s_dim), dtype=s_dtype)

        self.done_t = np.zeros(shape=size, dtype=bool)
        self.mu_t   = np.zeros(shape=size, dtype=np.float64) if store_mu else None

        if store_mu:
            self.data_tuple = (self.s_t, self.a_t, self.r_t, self.s_tp1, self.done_t, self.mu_t)
        else:
            self.data_tuple = (self.s_t, self.a_t, self.r_t, self.s_tp1, self.done_t)

    def __len__(self):
        return self.count
    
    def _add(self,
             s_t,
             a_t,
             r_t,
             s_tp1,
             a_tp1,
             done_t,
             mu_ =None):

        self.s_t[self.head]    = s_t
        self.a_t[self.head]    = a_t
        self.r_t[self.head]    = r_t
        self.s_tp1[self.head]  = s_tp1
        self.done_t[self.head] = done_t
        
        if self.store_mu:
            assert mu_t is not None
            self.mu_t[self.head] = mu_t
        
        self.head += 1
        if self.head == self.size:
            self.head = 0
        
    def add(self,
             s_t,
             a_t,
             r_t,
             s_tp1,
             a_tp1,
             done_t,
             mu_t=None):

        self._add(s_t, a_t, r_t, s_tp1, done_t, mu_t)
        if len(self) < self.size:
            self.count += 1

    def sample_contiguous_batch(self, batch_size):
        if self.size == len(self): 
            low  = self.head
            high = self.head + self.size - batch_size + 1
        else:
            low  = 0
            high = len(self)-batch_size+1

        idx = np.random.randint(low=low, high=high)

        ret_tuple = (np.take(data,
                             np.arange(idx, idx+batch_size),
                             axis=0,
                             mode='wrap'
                             ) for data in self.data_tuple
                     )

        return tuple(ret_tuple)

    def sample_dispersed_batch(self, batch_size):
        idx = np.random.randint(low=0, high=len(self), size=batch_size)

        if self.store_mu:
            return self.s_t[idx], self.a_t[idx], self.r_t[idx], self.s_tp1[idx], self.done_t[idx], self.mu_t[idx]
        else:
            return self.s_t[idx], self.a_t[idx], self.r_t[idx], self.s_tp1[idx], self.done_t[idx]

    def sample_batch(self, batch_size, contiguous=False):
        if len(self) < batch_size:
            Warning('The number of stored experience {} is less than batch_size {}.'.format(len(self), batch_size))
            batch_size = len(self)

        if contiguous:
            return self.sample_contiguous_batch(batch_size)
        else:
            return self.sample_dispersed_batch(batch_size)


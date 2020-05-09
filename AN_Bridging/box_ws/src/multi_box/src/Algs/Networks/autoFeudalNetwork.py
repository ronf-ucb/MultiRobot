#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from network import Network
from AutoEncoder import AutoEncoder
from feudalNetwork import Worker
import numpy as np

class AutoFeudalNet(nn.Module):
    def __init__(self, funPars):
        super(AutoFeudalNet, self).__init__()
        num_actions = funPars['u']
        s           = funPars['s']
        s_reduce    = funPars['s_reduce']
        s_workers   = funPars['s_workers']
        d           = funPars['d']
        k           = funPars['k']
        self.agents = 2
        self.children = [Worker(num_actions, s_workers[i], k, d) for i in range(self.agents)]
        self.parent = Parent(num_actions, s_reduce, d)
        self.encoder = nn.Linear(s, s_reduce)


    def forward(self, x, w_x, m_lstm, w_lstm, goals_horizon):
        x_encode = self.encoder(x)

        m_inputs = (x_encode, m_lstm)
        goals, m_lstm, m_value, m_state = self.parent(m_inputs)
        
        #Assumption: our starting goals summed together must be 0
        w_values_ext = [i for i in range(self.agents)]
        w_values_int = [i for i in range(self.agents)]
        policy       = [i for i in range(self.agents)]

        for i in range(self.agents):
            goals_horizon[i] = torch.cat([goals_horizon[i][:,1:], goals[i].unsqueeze(1)], dim=1)
            w_inputs = (w_x[i], w_lstm[i], goals_horizon[i])
            policy[i], w_lstm[i], w_values_ext[i], w_values_int[i] = self.children[i](w_inputs)

        return policy, goals, goals_horizon, m_lstm, w_lstm, m_value, w_values_ext, w_values_int, m_state


class Parent(nn.Module):
    def __init__(self, u_n, s_n, d):
        super(Parent, self).__init__()
        self.u_n = u_n
        self.s_n = s_n 
        self.d = d

        self.fc = nn.Linear(s_n, d)

        self.lstm = nn.LSTMCell(d, hidden_size= d)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc_critic1 = nn.Linear(d, 256)
        self.fc_critic2 = nn.Linear(256, 1)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.fc(x))
        state = x
        hx, cx = self.lstm(x, (hx, cx))

        goal = cx
        goals = []

        i = np.random.random()
        if i < .05: #generate random goal
            num = goal.numel()
            goal = torch.empty(num).normal_().view(goal.size())

        goals.append(goal)
        goals.append(goal)

        value = F.relu(self.fc_critic1(goal))
        value = self.fc_critic2(value)
        
        goal_norm = torch.norm(goal, p=2, dim=1).unsqueeze(1)
        goal = goal / goal_norm.detach()
        return goals, (hx, cx), value, state

        
    

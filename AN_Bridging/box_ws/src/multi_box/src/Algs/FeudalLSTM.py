#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.distributions import Normal
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from agent import Agent
from Replay import Replay

'''Original paper uses LSTMs. Will use feedforward for now.'''

class Feudal(Agent):
    def __init__(self, params, name, task):
        super(Feudal, self).__init__(params, name, task)
        if self.trainMode:
            self.wPars = params['wPars']
            self.wTrain = params['wTrain']
            self.mPars = params['mPars']
            self.mTrain = params['mTrain']
        else:
            pass
        self.worker   = Worker(self.wPars)
        self.manager  = Manager(self.mPars)

        self.perWidth = self.wPars['perception']
        self.perception = nn.Linear(self.n_in, self.perWidth)
        self.mSpace = nn.Linear(self.perWidth, self.perWidth)


        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()
    
    def forward(self, x, states):
        states_W, states_M, ss = states

    def saveModel(self):
        pass
        
    def train(self):
        pass

def reset_grad(t):
    no_grad = t.detach()
    return no_grad

class Manager(nn.Module):
    def __init__(self, pars):
        super(Manager, self).__init__()
        self.c = pars['c']
        self.d = pars['d']

        self.f_mspace = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.value    = nn.Linear(d, 1)
        self.f_Mrnn   = nn.LSTMCell(d, d)
    
    def init_state(self, batch_size):
        return (torch.zeros(batch_size, self.d), requires_grad = True,
                torch.zeros(batch_size, self.d), requires_grad = True)
    
    def forward(self, z, states_M):
        #z is state after pass into perception
        #states_M is previous hidden state in LSTM 

        s = self.f_mspace(z)
        g, _ = states_M = self.f_Mrnn(s, states_M)
        g = F.normalize(g)

        value = self.value(s)

        return value, g, s, states_M


class Worker(nn.Module):
    def __init__(self, pars):
        super(Worker, self).__init__()
        self.d   = pars['d']
        self.u_n = pars['u_n']
        self.k   = pars['k'] 
        self.hidden = self.u_n * self.k

        self.f_Wrnn = nn.LSTMCell(self.d, hidden)

        self.phi = nn.Linear(d, k, bias = False),

        self.value = nn.Linear(hidden, 1)

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden, requires_grad = True),
            torch.zeros(batch_size, self.hidden, requires_grad = True)
        )

    def forward(self, z, sum_g, states_W):
        w = self.phi(sum_g)
        w = w.view(-1, 1)

        U_flat, c = states_W = self.f_Wrnn(z, states_W)
        U = U_flat.view(self.u_n, self.k)

        a = torch.matmul(U, w)
        probs = F.softmax(a, dim = 1)
        value = self.value(z)

        return value, probs, states_W


            

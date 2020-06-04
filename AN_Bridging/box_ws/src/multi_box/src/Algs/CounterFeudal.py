#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import math 
import rospy
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from agent import Agent
from Networks.network import Network
from Networks.feudalNetwork import FeudalNetwork
from Buffers.CounterFeudalBuffer import Memory
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_state', 'local','goals'))


class CounterFeudal(object):
    def __init__(self, params, name, task):
        self.name           = name
        self.task           = task
        self.vTrain         = params['valTrain'] # Counterfactual network
        self.vPars          = params['valPars']
        self.aTrain         = params['actTrain'] # Local Actors
        self.aPars          = params['actPars']
        self.m_params       = params['m_pars'] # Manager
        self.m_train        = params['m_train']
        self.agents         = params['agents'] # Agents

        self.pubs = {}
        self.actionMap        = {0: (-2,-1), 1:(-1,-2), 2:(-2,-2), 3:(1,2), 4:(2,2), 5:(2,1), 6: (-2, 2), 7: (2, -2)} 
        for key in self.agents.keys():
            bot             = self.agents[key]
            self.pubs[key]  = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)
        rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)

        self.tau            = self.vPars['tau']
        self.int_weight     = self.vPars['int_weight']
        self.trainMode      = self.vPars['trainMode']
        self.batch_size     = self.vTrain['batch']
        self.td_lambda      = .8

        self.h_state_n      = self.aPars['u_n'] * self.aPars['k']
        self.c              = self.m_params['c']
        self.w_discount     = self.vTrain['gamma']
        self.m_discount     = self.m_train['gamma']
        self.clip_grad_norm = self.aTrain['clip']
        self.prevState      = None

        self.exp            = Memory()
        self.valueLoss      = []
        self.actorLoss      = []
        self.temp           = []
        self.goal_temp1     = None 
        self.goal_temp2     = None
        self.iteration      = 0
        self.totalSteps     = 0
        self.reward_manager = 0
        

        self.counter_critic = Network(self.vPars, self.vTrain)
        self.counter_target = Network(self.vPars, self.vTrain)
        self.manager = CounterManager(self.m_params, self.m_train) # manager
        self.actor = CounterActor(self.aPars, self.aTrain) # actor
        self.critic = CounterCritic(self.local_vPars, self.local_vTrain) 
        self.target = CounterCritic(self.local_vPars, self.local_vTrain)

        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.reset()

        task.initAgent(self)
        self.stop = False
        while(not self.stop):
            x = 1+1

        task.postTraining()

    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.task.restartProtocol(restart = 1)

    def get_action(self, s_true, s_split):
        if self.iteration % self.c == 0: 
            self.goal, value = self.manager(torch.FloatTensor(s_true)) # get a new goal
            self.temp_manager = m_Temp(self.goal, torch.FloatTensor(s_true), value)
        else: # Use goal transition 
            self.goal = self.prevState + self.goal - torch.FloatTensor(s_true)

        self.goal_temp2 = self.goal_temp1 
        self.goal_temp1 = self.goal

        policies = [self.actor(s, self.goal[i]) for i, s in enumerate(s_split)]       
        act_indices = [self.choose(policy) for policy in policies]
        actions = [self.actionMap[index] for index in act_indices]

        self.prevState = torch.FloatTensor(s_true)
        self.iteration += 1

        return np.array(actions), act_indices

    def choose(self, policies):
        m = Categorical(policies)
        action = m.sample()
        action = action.data.cpu().numpy()
        if action.size == 1:
            return np.asscalar(action)
        return torch.Tensor(action).unsqueeze(1)
    
    def saveModel(self):      
        pass

    def store(self, s, a, r, sprime, aprime, done, s_w):
        self.temp.append(Transition(s, a, r, sprime, s_w, self.goal_temp2))
        if self.iteration % self.c == 1 and self.iteration != 1: # remember, we push at 1 because we incremented in get_action
            self.exp.push(self.temp) # store into exp
            self.temp = []

    def reset(self):
        self.train(True)
        self.iteration = 0
        self.h = [torch.zeros(1, 1, self.h_state_n).to(device) for i in range(len(self.agents))]
        self.temp_first, self.temp_second = (None, None)
        return 

    def zipStack(self, data):
        data        = zip(*data)
        data        = [torch.stack(d).squeeze().to(device) for d in data]
        return data

    def train(self, episode_done = False): 
        if len(self.worker_exp) > self.batch_size:
            # UNPACK REPLAY
            # for each agent:
                # Extract manager samples by: taking first true state, taking first goal, sum rewards, last next state of each group
                # replace the goal: Sample new goals according to the paper 
                # for each of the goals, get goal transitions using worker state_transitions. 
                # Pass goal transitions concatenated with worker_states to get policy distribution
                # gather all actions according to the replay actions 
                # multiply probabilities across time
                # choose the goal index that has highest probability of all and replace each of the groups with the new goal and transition
            # Train counterfactual
                # Same as continuous counterfactual learning. Use montecarlo for each of the manager transitions 
            
            # Train manager 
                # Same as continuous counterfactual learning 
            
            # Train value
                # Same as normal value learning 
            
            # Train worker
                # Use advantage actor critic but through experience replay
            
            # store the groupings back into the replay
            print('yes')
        return 1


class CounterManager(nn.Module):
    def __init__(self, params, train):
        # define feed forward network here
        super(CounterManager, self).__init__()
        self.width      = params['width']
        self.x_state_n  = params['x_state_n']
        self.lr         = train['lr']
        self.mean       = params['mu']
        self.std        = params['std']

        self.fc1        = nn.Linear(self.x_state_n, self.width)
        self.fc2        = nn.Linear(self.width, self.width)
        self.fc3        = nn.Linear(self.width, self.x_state_n)

        self.optimizer  = optim.Adam(super(CounterManager, self).parameters(), lr=self.lr)
        return

    def preprocess(self, inputs):
        return (inputs - self.mean) / self.std
    
    def forward(self, s_true):
        inpt = self.preprocess(s_true) 
        inpt = F.leaky_relu(self.fc1(inpt))
        out  = F.leaky_relu(self.fc2(inpt))
        out  = self.fc3(out)
        out  = out / out.norm().detach()
        return out

class CounterActor(nn.Module):
    def __init__(self, params, train):
        super(CounterActor, self).__init__()
        self.h_state_n  = params['h_state_n']
        self.x_state_n  = params['x_state_n']
        self.u_n        = params['u_n']
        self.lr         = train['lr']
        self.mean       = params['mu']
        self.std        = params['std']

        self.fc1        = nn.Linear(self.x_state_n, self.h_state_n)
        self.fc2        = nn.Linear(self.h_state_n, self.h_state_n)
        self.fc3        = nn.Linear(self.h_state_n, self.u_n)
        self.soft       = nn.Softmax(dim = 1)
        self.optimizer  = optim.Adam(super(CounterActor, self).parameters(), lr=self.lr)

    def preprocess(self, inputs):
        return (inputs - self.mean) / self.std

    def forward(self, x):
        x = self.preprocess(x)
        inp = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(inp))
        out = self.fc3(out)
        policy = self.soft(out)
        return policy

    


    

#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import math 
import rospy
import time
import torch.nn.functional as F
from torch.distributions import Categorical
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from agent import Agent
from Networks.network import Network
from Networks.feudalNetwork import FeudalNetwork
from Buffers.CounterFactualBuffer import Memory
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Counter(object):
    def __init__(self, params, name, task):
        self.name           = name
        self.task           = task
        self.vTrain         = params['valTrain']
        self.vPars          = params['valPars']
        self.aTrain         = params['actTrain']
        self.aPars          = params['actPars']
        self.agents         = params['agents']

        self.pubs = {}
        # THIS IS A TEST
        self.actionMap      = {0: (-2,-1,-1), 1:(-1,-2,-1), 2:(-2,-2,-1), 3:(1,2,-1), 4:(2,2,-1), 
                                5:(2,1,-1), 6: (-2, 2, -1), 7: (2, -2, -1), 8:(0,0,-1)}
        for key in self.agents.keys():
            bot             = self.agents[key]
            self.pubs[key]  = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)
        rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)

        self.valueLoss      = []
        self.actorLoss      = []

        self.h_state_n      = self.aPars['h_state_n']
        self.x_state_n      = self.aPars['x_state_n']
        self.u_n            = self.aPars['u_n']
        self.clip_grad_norm = self.aTrain['clip']
        self.homogenous     = self.aPars['share_params']

        self.critic         = Network(self.vPars, self.vTrain).to(device)
        self.target         = Network(self.vPars, self.vTrain).to(device)
        if self.homogenous:
            self.actor      = CounterActor(self.aPars, self.aTrain).to(device) 
        else:
            self.actor      = [CounterActor(self.aPars, self.aTrain) for i in range(len(self.agents))]

        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param)

        self.clip_grad_norm = self.aTrain['clip']
        self.trainMode      = self.vPars['trainMode']
        self.batch_size     = self.vTrain['batch']
        self.discount       = self.vTrain['gamma']
        self.td_lambda      = .8
        self.tau            = .005
        self.stop           = False

        self.exp            = Memory()

        self.totalSteps     = 0

        self.reset()

        task.initAgent(self)

        while(not self.stop):
            x = 1+1

        task.postTraining()

    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.task.restartProtocol(restart = 1)

    def get_action(self, s_true, s_split):
        if self.homogenous:
            policy1 = self.actor(torch.FloatTensor(s_split[0]))
            a1 = np.asscalar(self.choose(policy1))
            policy2 = self.actor(torch.FloatTensor(s_split[1]))
            a2 = np.asscalar(self.choose(policy2))
        else:
            policy1 = self.actor[0](torch.FloatTensor(s_split[0]))
            a1 = self.choose(policy1)
            policy2 = self.actor[1](torch.FloatTensor(s_split[1]))
            a2 = self.choose(policy2)
        action = [self.actionMap[a1], self.actionMap[a2]]
        return np.array(action), [a1, a2]

    def choose(self, policies):
        m = Categorical(policies)
        action = m.sample()
        action = action.data.cpu().numpy()
        return action

    
    def saveModel(self):
        pass

    def store(self, s, a, r, sprime, aprime, done, s_w):
        self.exp.push(s, a, r, 1 - done, aprime, sprime, s_w)

    def reset(self):
        time.sleep(3)
        self.train(True)
        return 

    def zipStack(self, data):
        data        = zip(*data)
        data        = [torch.stack(d).squeeze().to(device) for d in data]
        return data

    def train(self, episode_done = False):
        if len(self.exp) > self.batch_size:
            transition = self.exp.sample(self.batch_size)
            states = torch.squeeze(torch.Tensor(transition.state)).to(device)
            states_next = torch.squeeze(torch.Tensor(transition.next_state)).to(device) 
            actions = torch.Tensor(transition.action).float().to(device)
            actions_next = torch.Tensor(transition.next_action).float().to(device) 
            rewards = torch.Tensor(transition.reward).to(device)
            masks = torch.Tensor(transition.mask).to(device)
            local = self.zipStack(transition.local)

            actions_next = []
            policies = []
            for s in local:
                policy = self.actor(s)
                policies.append(policy)
                a = self.choose(policy)
                actions_next.append(torch.FloatTensor(a))

            # Critic Update
            ID = torch.Tensor(states.size()[0], 1).fill_(0)
            inp = torch.cat((states_next, actions_next[1].unsqueeze(1), ID), dim = 1)
            q_tar = self.target(inp).detach().gather(1, actions_next[0].long().unsqueeze(1))
            q_tar = rewards.unsqueeze(1) + self.discount * masks.unsqueeze(1) * q_tar
            inp = torch.cat((states, actions[:, 1:], ID), dim = 1)
            q = self.critic(inp).gather(1, actions[:, 0].long().unsqueeze(1))
            loss = self.critic.get_loss(q, q_tar)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()

            ID = torch.Tensor(states.size()[0], 1).fill_(1)
            inp = torch.cat((states_next, actions_next[0].unsqueeze(1), ID), dim = 1)
            q_tar = self.target(inp).detach().gather(1, actions_next[1].long().unsqueeze(1))
            q_tar = rewards.unsqueeze(1) + self.discount * masks.unsqueeze(1) * q_tar
            inp = torch.cat((states, actions[:, :1], ID), dim = 1)
            q = self.critic(inp).gather(1, actions[:, 1].long().unsqueeze(1))
            loss = self.critic.get_loss(q, q_tar)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()

            
            actor_loss = 0
            # Actor Update 

            ID = torch.Tensor(states.size()[0], 1).fill_(0)
            inp = torch.cat((states, actions[:, 1:], ID), dim = 1)
            q_out = self.critic(inp) #batch x num_actions
            policy = policies[0] #batch x num_actions
            policy = torch.transpose(policy, 0, 1) #transpose
            mat = torch.mm(q_out, policy)
            baseline = torch.diagonal(mat, 0).detach() #diagonal elements are your baselines! 
            #gather the proper q_out elements
            q_taken = q_out.gather(1, actions[:, 0].long().unsqueeze(1))
            coma = (q_taken - baseline).detach()
            policy = torch.transpose(policy, 0, 1) #tranpose back
            probs_taken = policy.gather(1, actions[:, 0].long().unsqueeze(1))
            probs_taken[masks == 0] = 1.0
            loss = -(torch.log(probs_taken) * coma * masks).sum() / masks.sum()
            actor_loss += loss 

            ID = torch.Tensor(states.size()[0], 1).fill_(1)
            inp = torch.cat((states, actions[:, :1], ID), dim = 1)
            q_out = self.critic(inp) #batch x num_actions
            policy = policies[1] #batch x num_actions
            policy = torch.transpose(policy, 0, 1) #transpose
            mat = torch.mm(q_out, policy)
            baseline = torch.diagonal(mat, 0).detach() #diagonal elements are your baselines! 
            #gather the proper q_out elements
            q_taken = q_out.gather(1, actions[:, 1].long().unsqueeze(1))
            coma = (q_taken - baseline).detach()
            policy = torch.transpose(policy, 0, 1) #tranpose back
            probs_taken = policy.gather(1, actions[:, 1].long().unsqueeze(1))
            probs_taken[masks == 0] = 1.0
            loss = -(torch.log(probs_taken) * coma).mean()
            actor_loss += loss 


            self.actorLoss.append(actor_loss)

            if self.homogenous:
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            else:
                for actor in self.actor:
                    actor.optimizer.zero_grad()
                actor_loss.backward()
                for actor in self.actor:
                    actor.optimizer.step()    

            self.totalSteps += 1

            #UPDATE TARGET NETWORK:
            for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            return 

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
    

    
    


    

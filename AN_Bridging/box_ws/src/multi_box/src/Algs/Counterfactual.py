#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import math 
import rospy
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
        self.actionMap        = {0: (-2,-1), 1:(-1,-2), 2:(-2,-2), 3:(1,2), 4:(2,2), 5:(2,1), 6: (-2, 2), 7: (2, -2), 8:(0,0)} 
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

        self.critic         = Network(self.vPars, self.vTrain).to(device)
        self.target         = Network(self.vPars, self.vTrain).to(device)
        self.actor          = CounterActor(self.aPars, self.aTrain).to(device) 

        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param)

        self.clip_grad_norm = self.aTrain['clip']
        self.trainMode      = self.vPars['trainMode']
        self.step           = self.vTrain['step']
        self.discount       = self.vTrain['gamma']
        self.td_lambda      = .8
        self.tau            = .005
        self.stop           = False

        self.exp            = Memory()
        self.temp_first     = None
        self.temp_second    = None 

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
        policy1, h_new1 = self.actor(torch.FloatTensor(s_split[0]), self.h[0])
        a1 = self.choose(policy1)
        policy2, h_new2 = self.actor(torch.FloatTensor(s_split[1]), self.h[1])
        a2 = self.choose(policy2)
        self.temp_second = self.temp_first # due to implementation in previous 
        self.temp_first = [policy1, policy2]
        action = [self.actionMap[a1], self.actionMap[a2]]
        return np.array(action), [a1, a2]

    def choose(self, policies):
        m = Categorical(policies)
        action = m.sample()
        action = action.data.cpu().numpy()
        return np.asscalar(action)

    
    def saveModel(self):
        pass

    def store(self, s, a, r, sprime, aprime, done, s_w):
        self.exp.push(s, a, r, 1 - done, aprime, self.temp_second, sprime)

    def reset(self):
        self.h = [torch.zeros(1, self.h_state_n).to(device) for i in range(len(self.agents))]
        return 

    def get_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** (1. / 2)
        return grad_norm

    def get_lambda_targets(self, rewards, mask, gamma, target_qs):
        target_qs = target_qs.squeeze()
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[-1] = target_qs[-1] * mask[-1]


        for t in range(ret.size()[0] - 2, -1,  -1): #TODO: check that the mask is applied correctly
            ret[t] = self.td_lambda * gamma * ret[t + 1] + \
                mask[t] * (rewards[t] + (1 - self.td_lambda) * gamma * target_qs[t + 1])
        return ret.unsqueeze(1)

    def train(self):
        if len(self.exp) > self.step:
            transition = self.exp.sample()
            states = torch.squeeze(torch.Tensor(transition.state)).to(device)
            states_next = torch.squeeze(torch.Tensor(transition.next_state)).to(device) 
            actions = torch.Tensor(transition.action).float().to(device)
            actions_next = torch.Tensor(transition.next_action).float().to(device) 
            rewards = torch.Tensor(transition.reward).to(device)
            masks = torch.Tensor(transition.mask).to(device)
            policies = zip(*transition.policy)
            policies = [torch.stack(pol).squeeze().to(device) for pol in policies]


            for agent in range(len(self.agents)):
                #passing in true state, actions of all agents except current, agent ID
                ID = torch.Tensor(states.size()[0], 1).fill_(agent)
                inp = torch.cat((states_next, actions_next[:, :agent], actions_next[:, agent + 1:], ID), dim = 1)
                q_tar = self.target(inp).detach().gather(1, actions_next[:, agent].long().unsqueeze(1))
                q_tar = self.get_lambda_targets(rewards, masks, self.discount, q_tar)
                inp = torch.cat((states, actions[:, :agent], actions[:, agent + 1:], ID), dim = 1)
                q = self.critic(inp).gather(1, actions[:, agent].long().unsqueeze(1))
                loss = self.critic.get_loss(q, q_tar)
                self.critic.optimizer.zero_grad()
                loss.backward()
                self.critic.optimizer.step()
                #print('value loss: ', loss)
            self.valueLoss.append(loss)
            
            actor_loss = 0
            for agent in range(len(self.agents)):
                ID = torch.Tensor(states.size()[0], 1).fill_(agent)
                inp = torch.cat((states, actions[:, :agent], actions[:, agent + 1:], ID), dim = 1)
                q_out = self.critic(inp) #batch x num_actions
                policy = policies[agent] #batch x num_actions
                policy = torch.transpose(policy, 0, 1) #transpose
                mat = torch.mm(q_out, policy)
                baseline = torch.diagonal(mat, 0).detach() #diagonal elements are your baselines! 
                #gather the proper q_out elements
                q_taken = q_out.gather(1, actions[:, agent].long().unsqueeze(1))
                coma = (q_taken - baseline).detach()
                policy = torch.transpose(policy, 0, 1) #tranpose back
                probs_taken = policy.gather(1, actions[:, agent].long().unsqueeze(1))
                probs_taken[masks == 0] = 1.0
                loss = -(torch.log(probs_taken) * coma * masks).sum() / masks.sum()
                actor_loss += loss 
            #print('actor_loss', actor_loss)
            self.actorLoss.append(actor_loss)


            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
            self.actor.optimizer.step()
            self.exp = Memory()
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
        self.gru        = nn.GRU(self.h_state_n, self.h_state_n, num_layers = 1, batch_first = True)
        self.fc2        = nn.Linear(self.h_state_n, self.u_n)
        self.soft       = nn.Softmax(dim = 1)
        self.optimizer  = optim.Adam(super(CounterActor, self).parameters(), lr=self.lr)

    def preprocess(self, inputs):
        return (inputs - self.mean) / self.std

    def forward(self, x, h):
        #TODO: Implement epsilon lower-bounds
        x = self.preprocess(x)
        inp = self.fc1(x)

        inp = torch.unsqueeze(inp, 0)
        h = torch.unsqueeze(h, 0)

        out, h_new = self.gru(inp, h)
        out = self.fc2(out)
        out_n = out.size()
        out = out.view(out_n[0], out_n[2])
        policy = self.soft(out)
        return policy, h_new
    

    
    


    

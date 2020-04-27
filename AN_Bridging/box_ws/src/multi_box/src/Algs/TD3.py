#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from Networks.network import Network
from Networks.TD3Network import TD3Network
from agent import Agent
from utils import positiveWeightSampling as priority
from utils import OUNoise
from Replay import Replay

'''Twin-delayed DDPG to curb Q value overestimation with clipped double Q-learning, Q value smoothing using noise and delayed policy updates for stability'''

class Twin_DDPG(Agent):
    def __init__(self, params, name, task):
        super(Twin_DDPG, self).__init__(params, name, task)
        if self.trainMode:
            self.values = [Network(self.vPars, self.vTrain), Network(self.vPars, self.vTrain)]
            self.tar = [Network(self.vPars, self.vTrain), Network(self.vPars, self.vTrain)]
            for i in range(len(self.values)):
                self.tar[i].load_state_dict(self.values[i].state_dict())
            self.aPars = params['actPars']
            self.aTrain = params['actTrain']
            self.policyNet = TD3Network(self.aPars, self.aTrain)
            self.tarPolicy = TD3Network(self.aPars, self.aTrain)
            self.tarPolicy.load_state_dict(self.policyNet.state_dict())
        else:
            self.policyNet = Network(self.aPars, self.aTrain)
            self.policyNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/PolicyNet.txt"))

        self.base = self.vTrain['baseExplore']
        self.step = self.vTrain['step']
        self.expSize =self.vTrain['buffer']
        self.exp = Replay(self.expSize)
        self.a = self.vTrain['a']
        self.tau = self.vPars['tau']
        self.smooth = self.vTrain['smooth']
        self.clip = self.vTrain['clip']
        self.delay = self.vTrain['policy_delay']
        self.out_n = self.aPars['out_n']
        self.mean_range = self.aPars['mean_range']
        self.noise = OUNoise(self.out_n, mu = 0, theta = .20, max_sigma = self.explore, min_sigma = self.base, decay_period = self.step)
        self.replaceCounter = 0
        self.valueLoss = []
        self.actorLoss = []
        self.avgLoss = 0
        self.avgActLoss = 0

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()
    
    def saveModel(self):
        torch.save(self.policyNet.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/TD3_goal_policy.txt")
        torch.save(self.values[0].state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/TD3_goal_" + "Qvalue1" + ".txt")
        torch.save(self.values[1].state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/TD3_goal_" + "Qvalue2" + ".txt")
        print("Network saved")
        pass
        
    def train(self):
        if self.dataSize > self.batch_size:
            self.totalSteps += 1
            s, a, r, n_s, n_a, mask = self.exp.get_data()
            mask = torch.FloatTensor(np.where(mask  > .5, 0, 1)) #if fail, equal to 1 so set mask to 0

            n_a = self.tarPolicy(torch.FloatTensor(n_s)).detach().numpy() 

            #target policy smoothing 
            n_a_ = n_a + np.clip(np.random.normal(0, self.smooth, n_a.shape), -self.clip, self.clip)
            n_sa = torch.FloatTensor(np.hstack((n_s, n_a))).detach() 
            qtar = torch.FloatTensor(r) + self.discount*mask*torch.min(self.tar[0](n_sa).detach(), self.tar[1](n_sa).detach()) #pass in

            #priority sampling 
            q = (sum([self.values[i](torch.FloatTensor(np.hstack((s, a)))) for i in range(len(self.values))])/(len(self.values))).detach()
            c = np.random.choice(min(self.dataSize, self.expSize), self.batch_size, priority(np.abs(q-qtar).numpy() + 1e-9, self.a))

            #policy update
            if self.trainIt % self.delay == 0:
                act = self.policyNet(torch.FloatTensor(s[c]))
                s_a = torch.cat((torch.FloatTensor(s[c]), act), 1)
                q = self.values[0](s_a)
                policy_loss = -q.mean()

                self.policyNet.optimizer.zero_grad()
                policy_loss.backward()
                self.policyNet.optimizer.step()
                self.policyNet.scheduler.step()
                self.avgActLoss += policy_loss

            sa = torch.FloatTensor(np.hstack((s[c], a[c])))
            for i in range(len(self.values)):
                loss = self.values[i].loss_fnc(self.values[i](sa), qtar[c])
                self.values[i].optimizer.zero_grad()
                loss.backward()
                self.values[i].optimizer.step()
                self.values[i].scheduler.step()
                self.avgLoss += loss / len(self.values) 

            #iteration updates
            self.trainIt += 1

            for target_param, param in zip(self.tarPolicy.parameters(), self.policyNet.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
            for i in range(len(self.values)):
                for target_param, param in zip(self.tar[i].parameters(), self.values[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            

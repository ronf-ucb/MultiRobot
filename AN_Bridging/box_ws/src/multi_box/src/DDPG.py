#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt
from agent import Agent
from utils import *

class DDPG(Agent):
    def __init__(self, params, name, task):
        super(DDPG, self).__init__(params, name, task)
        if self.trainMode:
            self.tarNet = Network(self.vPars, self.vTrain)
            self.aPars = params['actPars']
            self.aTrain = params['actTrain']
            self.policyNet = Network(self.aPars, self.aTrain)
            self.tarPolicy = Network(self.aPars, self.aTrain)
            self.tarNet.load_state_dict(self.valueNet.state_dict())
            self.tarPolicy.load_state_dict(self.policyNet.state_dict())
        else:
            self.valueNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNet.txt"))
            self.policyNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/PolicyNet.txt"))

        self.base = self.vTrain['baseExplore']
        self.decay = self.vTrain['decay']
        self.step = self.vTrain['step']
        self.expSize =self.vTrain['buffer']
        self.exp = np.zeros((self.expSize, self.vTrain['replayDim']))
        self.priority = self.vTrain['prioritySample']
        self.a = self.vTrain['a']
        self.tau = self.vPars['tau']
        self.replaceCounter = 0
        self.valueLoss = []
        self.actorLoss = []
        self.avgLoss = 0
        self.avgActLoss = 0
        self.lrStep = 0
        self.state_n = self.state_n - self.u_n

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()
    

    def store(self, s, a, r, sprime, aprime = None, failure = 0):
        a = a.detach()
        self.exp[self.dataSize % self.expSize] = np.hstack((s, a, r, sprime))
    
    def saveModel(self):
        #torch.save(self.valueNet.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNet_box.txt")
        #print("Network saved")
        pass
        
    def train(self):
        if self.dataSize > self.batch_size:
            #probs = positiveWeightSampling(self.state_n + 1)
            idx = min(self.dataSize, self.expSize)
            states = self.exp[:idx, :self.state_n]
            actions = self.exp[:idx, self.state_n: self.state_n + self.u_n]
            rewards = self.exp[:idx, self.state_n + self.u_n: self.state_n + self.u_n + 1]
            nextStates = self.exp[:idx, -self.state_n:]

            #value update
            in_critic = torch.FloatTensor(np.hstack((states, actions)))
            q_critic = self.valueNet(in_critic) #pass in. Processing implied
            next_a_critic = self.tarPolicy(torch.FloatTensor(nextStates)).detach().numpy()
            in_critic = torch.FloatTensor(np.hstack((nextStates, next_a_critic))) 
            qtar = torch.FloatTensor(rewards) + self.discount*self.tarNet(in_critic).detach() #pass in

            if self.priority:
                qcopy = q_critic.clone()
                delta = np.abs((qtar - qcopy.detach()).numpy())
                delta = delta + 1e-8*np.ones(delta.shape)
                choices = np.random.choice(min(self.dataSize, self.expSize) , self.batch_size, positiveWeightSampling(delta, self.a))
            else:
                choices = np.random.choice(min(self.dataSize, self.expSize), self.batch_size) 

            #policy update
            newAct = self.policyNet(torch.FloatTensor(states[choices]))
            polIn = torch.cat((torch.FloatTensor(states[choices]), newAct), 1)
            q_policy = self.valueNet(polIn)
            policy_loss = -(torch.sum(q_policy)/self.batch_size) #chain rule gets applied!

            q_critic = q_critic[choices]
            qtar = qtar[choices]
            critic_loss = self.valueNet.loss_fnc(q_critic, qtar) / self.batch_size

            self.policyNet.optimizer.zero_grad()
            policy_loss.backward()
            self.policyNet.optimizer.step()
            self.avgActLoss += policy_loss
            
            self.valueNet.optimizer.zero_grad()
            critic_loss.backward()
            self.valueNet.optimizer.step()
            self.avgLoss += critic_loss/self.batch_size 

            self.trainIt += 1
            self.lrStep += 1

            if self.replaceCounter % 200 == 0:
                self.tarNet.load_state_dict(self.valueNet.state_dict())
                self.tarPolicy.load_state_dict(self.policyNet.state_dict())
                self.replaceCounter = 0
            self.replaceCounter += 1

            if self.lrStep % self.step == 0:
                newExplore = (self.explore[0] - self.base)*self.decay + self.base
                self.explore = (newExplore, self.explore[1])
                print(" ############# NEW EPSILON: ", self.explore, " ################")
            

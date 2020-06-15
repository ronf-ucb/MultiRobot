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
from agent import Agent
from Networks.dualNetwork import DualNetwork
from utils import positiveWeightSampling
from Buffers.CounterFactualBuffer import Memory

'''Double DQN with priority sampling based on TD error. Possible to have dual networks for advantage and value'''

class DoubleQ(Agent):
    def __init__(self, params, name, task):
        super(DoubleQ, self).__init__(params, name, task)
        self.dual = self.vPars['dual']
        if self.trainMode:
            if self.dual:
                self.tarNet = DualNetwork(self.vPars, self.vTrain)
                self.valueNet = DualNetwork(self.vPars, self.vTrain)
            else:
                self.tarNet = Network(self.vPars, self.vTrain)
                self.valueNet = Network(self.vPars, self.vTrain)
        else:
            self.valueNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork.txt"))

        self.expSize =self.vTrain['buffer']
        self.out_n = self.vPars['neurons'][-1]
        self.exp = Memory()
        self.double = self.vTrain['double']
        self.replaceCounter = 0
        self.valueLoss = []
        self.avgLoss = 0

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()

    def saveModel(self):
        pass
    
    def store(self, s, a, r, sprime, aprime, done):
        self.exp.push(s, a, r, 1-done, aprime, sprime)
    
    def get_action(self, s):
        q = self.valueNet(torch.FloatTensor(s))
        i = np.random.random()
        if i < self.explore:
            index = np.random.randint(self.out_n)
        else:
            q = q.detach().numpy()
            index = np.argmax(q)
        self.explore = max(.1, self.explore * .999)
        return index

    def train(self):
        if len(self.exp) > self.batch_size:
            states, actions, rewards, masks, _ , nextStates, _,_,_ = self.exp.sample(batch = self.batch_size)

            if self.replaceCounter % 200 == 0:
                self.tarNet.load_state_dict(self.valueNet.state_dict())
                self.replaceCounter = 0

            qValues = self.valueNet(torch.FloatTensor(states)).squeeze(1) #pass in. Processing implied
            q = torch.gather(qValues, 1, torch.LongTensor(actions).unsqueeze(1)) #get q values of actions  
            qnext = self.tarNet(torch.FloatTensor(nextStates)).squeeze(1).detach() #pass in

            if self.double: 
                qnextDouble = self.valueNet(torch.FloatTensor(nextStates)).squeeze(1).detach() #pass in
                qnext = torch.gather(qnext, 1, torch.LongTensor(qnextDouble.argmax(1).unsqueeze(1)))
                qtar = torch.FloatTensor(rewards).squeeze(1) + self.discount * torch.Tensor(masks).unsqueeze(1) * qnext
            else:
                qtar = torch.FloatTensor(rewards) + self.discount * torch.Tensor(masks).unsqueeze(1) * qnext.max(1)[0].view(self.batch_size, 1) #calculate target

            loss = self.valueNet.get_loss(q, qtar)
            self.valueNet.optimizer.zero_grad()
            loss.backward()
            self.valueNet.optimizer.step()
            self.avgLoss += loss
            self.replaceCounter += 1
            self.totalSteps += 1
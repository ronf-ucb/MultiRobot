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
from Replay import Replay

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

        self.base = self.vTrain['baseExplore']
        self.decay = self.vTrain['decay']
        self.out_n = self.vPars['out_n']
        self.step = self.vTrain['step']
        self.expSize =self.vTrain['buffer']
        self.exp = Replay(self.expSize)
        self.double = self.vTrain['double']
        self.priority = self.vTrain['prioritySample']
        self.a = self.vTrain['a']
        self.replaceCounter = 0
        self.valueLoss = []
        self.avgLoss = 0
        self.lrStep = 0

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()

    def saveModel(self):
        #torch.save(self.valueNet.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNet_box.txt")
        #print("Network saved")
        pass
        
    def train(self):
        self.totalSteps += 1
        states, actions, rewards, nextStates, dummy, dummy2 = self.exp.get_data()

        if self.replaceCounter % 200 == 0:
            self.tarNet.load_state_dict(self.valueNet.state_dict())
            self.replaceCounter = 0
        self.replaceCounter += 1
        #PREPROCESS AND POSTPROCESS SINCE WE ARE USING PREDICT TO SEND ACTIONS
        qValues = self.valueNet(torch.FloatTensor(states)) #pass in. Processing implied
        q = torch.gather(qValues, 1, torch.LongTensor(actions)) #get q values of actions  
        qnext = self.tarNet(torch.FloatTensor(nextStates)).detach() #pass in

        if self.double: 
            qnextDouble = self.valueNet(torch.FloatTensor(nextStates)).detach() #pass in
            qnext = torch.gather(qnext, 1, torch.LongTensor(qnextDouble.argmax(1).view(-1, 1)))
            qtar = torch.FloatTensor(rewards) + self.discount * qnext 
        else:
            qtar = torch.FloatTensor(rewards) + self.discount * qnext.max(1)[0].view(self.batch_size, 1) #calculate target

        if self.priority:
            qcopy = q.clone()
            delta = np.abs((qtar - qcopy.detach()).numpy())
            delta = delta + 1e-8*np.ones(delta.shape)
            choices = np.random.choice(min(self.dataSize, self.expSize) , self.batch_size, positiveWeightSampling(delta, self.a))
        else:
            choices = np.random.choice(min(self.dataSize, self.expSize), self.batch_size) 

        q = q[choices]
        qtar = qtar[choices]

        loss = self.valueNet.loss_fnc(q, qtar)
        self.valueNet.optimizer.zero_grad()
        loss.backward()
        self.valueNet.optimizer.step()
        self.avgLoss += loss
        self.trainIt += 1
        self.lrStep += 1
        if self.lrStep % self.step == 0:
            self.explore = (self.explore - self.base)*self.decay + self.base
            print(" ############# NEW EPSILON: ", self.explore, " ################")
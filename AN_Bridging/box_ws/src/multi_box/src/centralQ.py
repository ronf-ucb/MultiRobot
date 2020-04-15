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
from utils import positiveWeightSampling

class CentralQ(Agent):
    def __init__(self, params, name, task):
        super(CentralQ, self).__init__(params, name, task)
        if self.trainMode:
            self.tarNet = Network(self.vPars, self.vTrain)
        else:
            self.valueNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork.txt"))

        self.base = self.vTrain['baseExplore']
        self.decay = self.vTrain['decay']
        self.step = self.vTrain['step']
        self.expSize =self.vTrain['buffer']
        self.exp = np.zeros((self.expSize, self.vTrain['replayDim']))
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
    

    def store(self, s, a, r, sprime, aprime = None, failure = 0):
        self.exp[self.dataSize % self.expSize] = np.hstack((s, a, r, sprime))
    
    def saveModel(self):
        torch.save(self.valueNet.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNet_box.txt")
        print("Network saved")
        
    def train(self):
        if self.dataSize > self.batch_size:
            #probs = positiveWeightSampling(self.state_n + 1)
            idx = min(self.dataSize, self.expSize)
            states = self.exp[:idx, :self.state_n]
            actions = self.exp[:idx, self.state_n: self.state_n + 1]
            rewards = self.exp[:idx, self.state_n + 1: self.state_n + 2]
            nextStates = self.exp[:idx, -self.state_n:]
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
            self.avgLoss += loss/self.batch_size
            self.trainIt += 1
            self.lrStep += 1
            if self.lrStep % self.step == 0:
                self.explore = (self.explore - self.base)*self.decay + self.base
                print(" ############# NEW EPSILON: ", self.explore, " ################")
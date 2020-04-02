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

class CentralQ(Agent):
    def __init__(self, params):
        super(CentralQ, self).__init__(params)

        self.trainMode = self.valueParams['trainMode']
        if self.trainMode:
            self.QNetwork = self.valueNet
            self.targetNetwork = Network(self.valueParams, self.valueTrain)
        else:
            self.valueNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork.txt"))

        self.discount = self.valueTrain['gamma']
        self.weight_loc = self.valueTrain['alpha1']
        self.weight_vel = self.valueTrain['alpha2']
        self.weight_ori = self.valueTrain['alpha3']
        self.weight_agents = self.valueTrain['lambda']
        self.expSize = self.valueTrain['buffer']
        self.exploration = self.valueTrain['explore']

        self.replayFeatures = 2*self.state_n + 1 + 1 
        self.experience = np.zeros((self.expSize, self.replayFeatures))
        self.tankPub = rospy.Publisher(self.ROSParams["tankPub"], Vector3, queue_size = 1)
        self.bridgePub = self.aPub

        self.sigmoid = nn.Sigmoid()
        self.actorLoss = []
        self.own_n = 7

        self.u_n = 3
        self.replaceCounter = 0
        self.QLearner = True

        self.map = {0: -2, 1: 0, 2: 2}
    
        while(not self.stop):
            x = 1+1
        
        self.plotLoss(True, "Centralized Q Networks: Q Value Loss over Iterations")
        self.saveModel()
        
    def manageStatus(self, fail):
        if fail == 1:
            self.prevState = None 
            self.prevAction = None
            self.goalPosition = 0
            self.startDistance = []
            if self.trainIt > 0:
                self.valueLoss.append((self.avgLoss)/self.trainIt)
            self.avgLoss = 0    
            self.trainIt = 0
        else:
            self.fail = False

    def receiveState(self, message):
        floats = vrep.simxUnpackFloats(message.data)
        self.goalPosition = np.array(floats[-4:-1])
        failure = floats[-1]
        state = (np.array(floats[:self.state_n])).reshape(1,-1)
        if len(self.startDistance) == 0:
            for i in range(self.agents_n):
                pos = state[:, 3 + 4*i:6+4*i].ravel()
                self.startDistance.append(np.sqrt(np.sum(np.square(pos - self.goalPosition))))
        for i in range(self.agents_n - 1):
            '''TODO: receive the observations and ADD GAUSSIAN NOISE HERE PROPORTIONAL TO DISTANCE. Concatenate to the state'''
        index, ropeAction = (self.sendAction(state))
        if type(self.prevState) == np.ndarray:
            r = np.array(self.rewardFunction(state, ropeAction)).reshape(1,-1)
            self.experience[self.dataSize % self.expSize] = np.hstack((self.prevState, self.prevAction[:, :1], r, state))
            self.dataSize += 1
        self.prevState = state
        self.prevAction = index
        if self.trainMode:
            self.train()
        self.manageStatus(failure)
        return 
    
    def saveModel(self):
        torch.save(self.QNetwork.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork2.txt")
        print("Network saved")

    def sendAction(self, state):
        q = self.QNetwork.predict(state)
        i = np.random.random()
        if i < self.exploration:
            index = np.random.randint(84)
        else:
            index = np.argmax(q.detach().numpy())
        tank, bridge = self.index_to_action(index)
        self.tankPub.publish(tank)
        self.bridgePub.publish(bridge)
        return np.array([index]).reshape(1,-1), np.array([tank.z]).reshape(1,-1)

    def index_to_action(self, index):
        tankmsg = Vector3()
        bridgemsg = Vector3()
        if index >= 81:
            tankmsg.x = 0
            tankmsg.y = 0
            tankmsg.z = index - 81
            bridgemsg.x = 0
            tankmsg.y = 0
        else:
            tankmsg.x = self.map[index % 3]
            tankmsg.y = self.map[(index // 3) % 3]
            tankmsg.z = -1
            bridgemsg.x  = self.map[(index // 9) % 3]
            bridgemsg.y = self.map[(index//27) % 3]
        return (tankmsg, bridgemsg)
        
    def train(self):
        if self.dataSize > self.batch_size:
            choices = np.random.choice(min(self.dataSize, self.expSize), self.batch_size)
            data = self.experience[choices]
            states = data[:, :self.state_n]
            actions = data[:, self.state_n: self.state_n + 1]
            rewards = data[: self.state_n + 1: self.state_n + 2]
            nextStates = data[:, -self.state_n:]

            if self.replaceCounter % 100 == 0:
                self.targetNetwork.load_state_dict(self.QNetwork.state_dict())
            self.replaceCounter += 1
        
            q = self.QNetwork(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions))
            qnext = self.targetNetwork(torch.FloatTensor(nextStates)).detach()
            qtar = torch.FloatTensor(rewards) + self.discount * qnext.max(1)[0].view(self.batch_size, 1)
            loss = self.QNetwork.loss_fnc(q, qtar)

            self.QNetwork.optimizer.zero_grad()
            loss.backward()
            self.QNetwork.optimizer.step()
            self.avgLoss += loss/self.batch_size
            self.trainIt += 1
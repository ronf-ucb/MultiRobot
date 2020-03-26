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

        self.QNetwork = self.valueNet
        self.targetNetwork = Network(self.valueParams, self.valueTrain)

        self.discount = self.valueTrain['gamma']
        self.weight_loc = self.valueTrain['alpha1']
        self.weight_vel = self.valueTrain['alpha2']
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
    
        while(True):
            x = 1+1

    def rewardFunction(self, state, action):
        if self.fail:
            return -10

        state = state.ravel()
        prevState = self.prevState.ravel()
        robState = state[:self.state_n]
        position = np.array(state[3:6])
        prevPosition = np.array(prevState[3:6])

        '''Calculate distance from the goal location'''
        deltas = self.goalPosition - position
        R_loc = (1/np.sqrt(np.sum(np.square(deltas*deltas)))) * self.startDistance
        
        '''Calculate velocity along direction towards goal position '''
        deltas = position - prevPosition
        vector = self.goalPosition - prevPosition
        norm = np.sqrt(np.sum(np.square(vector)))
        vector = vector / norm
        R_vel = np.sum(deltas * vector)

        R_agents = 0
        for i in range(self.agents_n - 1):
            startIndex = self.own_n + 4*i
            position = state[startIndex: startIndex + 3]
            prevPosition = np.array(prevState[startIndex: startIndex+3])
            deltas = self.goalPosition - position

            R_agents += (1/np.sqrt(np.sum(np.square(deltas*deltas)))) * self.startDistance

            deltas = position - prevPosition
            vector = self.goalPosition - prevPosition
            norm = np.sqrt(np.sum(np.square(vector)))
            vector = vector / norm

            R_agents += np.sum(deltas * vector) #dot product  
        R_rope = -5 if ((self.u_n == 3) and (np.ravel(action)[-1] in self.ropes)) else 0

        reward = self.weight_loc * R_loc + self.weight_vel * R_vel + self.weight_agents * R_agents + R_rope
        return reward

    def receiveState(self, message):
        floats = vrep.simxUnpackFloats(message.data)
        self.goalPosition = np.array(floats[-3:])
        state = (np.array(floats[:self.state_n])).reshape(1,-1)
        if self.startDistance == 0:
            pos = state[:, 3:6].ravel()
            self.startDistance = np.sqrt(np.sum(np.square(pos - self.goalPosition)))
        for i in range(self.agents_n - 1):
            '''TODO: receive the observations and ADD GAUSSIAN NOISE HERE PROPORTIONAL TO DISTANCE. Concatenate to the state'''
        action = (self.sendAction(state)).reshape(1,-1)
        if type(self.prevState) == np.ndarray:
            r = np.array(self.rewardFunction(state, action)).reshape(1,-1)
            self.experience[self.dataSize % self.expSize] = np.hstack((self.prevState, self.prevAction[:, :1], r, state))
            self.dataSize += 1
        self.prevState = state
        self.prevAction = action
        self.train()
        return 

    def sendAction(self, state):
        q = self.QNetwork.predict(state)
        i = np.random.random()
        if i < self.exploration:
            index = np.random.randint(84)
        else:
            index = np.argmax(q.detach().numpy())
        if index >= 81:
            msg = Vector3()
            index = index - 81
            msg.x = 0
            msg.y = 0
            msg.z = index
            self.tankPub.publish(msg)
            self.bridgePub.publish(msg)
        else:
            tankLeft = index % 3
            tankRight = (index // 3) % 3
            bridgeLeft = (index // 9) % 3
            bridgeRight = (index // 27) % 3
            msg = Vector3()
            msg.x = self.map[tankLeft]
            msg.y = self.map[tankRight]
            msg.z = -1
            self.tankPub.publish(msg)
            msg.x = self.map[bridgeLeft]
            msg.y = self.map[bridgeRight]
            self.bridgePub.publish(msg)
        return np.array([index])
        


    def train(self):
        if self.dataSize > self.batch_size:
            choices = np.random.choice(min(self.dataSize, self.expSize), self.batch_size)
            data = self.experience[choices]
            states = data[:, :self.state_n]
            actions = data[:, self.state_n: self.state_n + 1]
            rewards = data[: self.state_n + 1: self.state_n + 2]
            nextStates = data[:, -self.state_n:]

            if self.replaceCounter % 500 == 0:
                self.targetNetwork.load_state_dict(self.QNetwork.state_dict())
            self.replaceCounter += 1
        
            q = self.QNetwork(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions))
            qnext = self.targetNetwork(torch.FloatTensor(nextStates)).detach()
            qtar = torch.FloatTensor(rewards) + self.discount * qnext.max(1)[0].view(self.batch_size, 1)
            loss = self.QNetwork.loss_fnc(q, qtar)

            self.QNetwork.optimizer.zero_grad()
            loss.backward()
            self.QNetwork.optimizer.step()
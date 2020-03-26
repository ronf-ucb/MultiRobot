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

# Collaborative agent in multi-agent framework. Initiates a class that contains:
    # actor network
    # critic network
    # actor network copy for KL divergence calculations

class Agent():
    def __init__(self, params, alg):
        self.alg = alg
        valueParams = params['valueParams']
        valueTrain = params['valueTrain']
        ROSParams = params['ROS']

        self.critic = Network(criticParams, criticTrain)

        stateSub = ROSParams['stateSub']
        subQ = ROSParams['subQueue']
        self.actionPub = ROSParams['actionPub']
        pubQ = ROSParams['pubQueue']
        self.agents_n = ROSParams['numAgents']
        self.delta_t = ROSParams['delta_t']
        self.vrep_sub = rospy.Subscriber("/failure", Int8, self.receiveStatus, queue_size = 1)
        self.aPub = rospy.Publisher(self.actionPub, Vector3, queue_size = pubQ)
        rospy.Subscriber(stateSub, String, self.receiveState, queue_size = subQ) 

        self.batch_size = valueTrain['batch']
        self.state_n = valueParams['state_n']

        replayFeatures = 2*self.state_n + self.u_n + 1 
        self.experience = np.zeros((self.expSize, replayFeatures))
        self.dataSize = 0 #number of data tuples we have accumulated so far
        self.sigmoid = nn.Sigmoid()

        self.prevState = None
        self.prevAction = None 

        self.goalPosition = 0 
        self.startDistance = 0
        self.fail = False
        self.ropes = [0,1,2]

        self.valueLoss = []
        
    def receiveStatus(self, message):
        if message.data == 1:
            self.prevState = None 
            self.prevAction = None
            self.goalPosition = 0
            self.startDistance = 0 
        else:
            self.fail = False
        
    def receiveState(self, message):
        floats = vrep.simxUnpackFloats(message.data)
        self.goalPosition = np.array(floats[-3:])
        state = (np.array(floats[:self.state_n])).reshape(1,-1)
        if self.startDistance == 0:
            pos = state[:, 3:6].ravel()
            self.startDistance = np.sqrt(np.sum(np.square(pos - self.goalPosition)))
        for i in range(self.agents_n - 1):
            '''TODO: receive the observations and ADD GAUSSIAN NOISE HERE PROPORTIONAL TO DISTANCE. Concatenate to the state'''
        action = self.sendAction(state)
        if type(self.prevState) == np.ndarray:
            r = np.array(self.rewardFunction(state, action)).reshape(1,-1)
            self.experience[self.dataSize % self.expSize] = np.hstack((self.prevState, self.prevAction, r, state))
            self.dataSize += 1
        self.prevState = state
        self.prevAction = action.reshape(1,-1)
        self.train()
        return 
        
    def sendAction(self, state):
        out = self.actor.predict(state)
        mean = (out.narrow(1, 0, self.u_n).detach()).numpy().ravel()
        msg = Vector3()
        if self.prob:
            var = (out.narrow(1, self.u_n, self.u_n)).detach().numpy().ravel()
            action = self.sample(mean, var) + np.random.normal(0, self.exploration, self.u_n)
            action += np.random.normal(0, self.exploration, self.u_n)
            msg.x = action[0]
            msg.y = action[1]
            if (self.u_n > 2):
                action[2] = np.round(action[2])
                msg.z = action[2]
            self.aPub.publish(msg)
        else:
            i = np.random.random()
            if i < self.exploration:
                msg.x = np.randint(-4, 5, size = 1)
                msg.y = np.randint(-4, 5, size = 1)
                if (self.u_n > 2):
                    msg.z = np.randint(0, 3, size = 1)
            else:
                msg.x = mean[0]
                msg.y = mean[1]
                msg.z = np.round(mean[2])           
        return action

    def sample(self, mean, var):
        return np.random.normal(mean, var)
    
    def plotLoss(self):
        plt.plot(range(len(self.criticLoss)), self.criticLoss)
        plt.title("Critic Loss over Iterations")
        plt.show()
        plt.plot(range(len(self.actorLoss)), self.actorLoss)
        plt.title("Actor Loss over Iterations")
        plt.show()

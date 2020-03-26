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

class Agent(object):
    def __init__(self, params):
        self.valueParams = params['valueParams']
        self.valueTrain = params['valueTrain']
        self.ROSParams = params['ROS']

        self.valueNet = Network(self.valueParams, self.valueTrain)

        stateSub = self.ROSParams['stateSub']
        subQ = self.ROSParams['subQueue']
        self.actionPub = self.ROSParams['actionPub']
        pubQ = self.ROSParams['pubQueue']
        self.agents_n = self.ROSParams['numAgents']
        self.delta_t = self.ROSParams['delta_t']
        self.vrep_sub = rospy.Subscriber("/failure", Int8, self.receiveStatus, queue_size = 1)
        self.aPub = rospy.Publisher(self.actionPub, Vector3, queue_size = pubQ)
        rospy.Subscriber(stateSub, String, self.receiveState, queue_size = subQ) 

        self.batch_size = self.valueTrain['batch']
        self.state_n = self.valueParams['state_n']

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
           
    def sendAction(self, state):
        out = self.actor.predict(state)
        mean = (out.narrow(1, 0, self.u_n).detach()).numpy().ravel()
        msg = Vector3()
        if self.prob: #probabilistic policy
            var = (out.narrow(1, self.u_n, self.u_n)).detach().numpy().ravel()
            action = self.sample(mean, var) + np.random.normal(0, self.exploration, self.u_n)
            action += np.random.normal(0, self.exploration, self.u_n)
            msg.x = action[0]
            msg.y = action[1]
            if (self.u_n > 2):
                action[2] = np.round(action[2])
                msg.z = action[2]
        else: #assume deterministic policy
            i = np.random.random()
            if i < self.exploration: 
                action = mean
                msg.x = mean[0]
                msg.y = mean[1]
                if (self.u_n > 2):
                    msg.z = np.round(mean[2])  
            else:
                if self.u_n > 2:
                    msg.x = np.random.randint(-4, 5, size = 1)
                    msg.y = np.random.randint(-4, 5, size = 1)
                if (self.u_n > 2):
                    msg.z = np.randint(0, 3, size = 1)
                action = np.array([msg.x, msg.y, msg.z])
        self.aPub.publish(msg)         
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

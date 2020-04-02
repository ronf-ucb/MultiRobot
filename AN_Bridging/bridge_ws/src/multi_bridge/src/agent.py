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
        self.finish_sub = rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)
        self.aPub = rospy.Publisher(self.actionPub, Vector3, queue_size = pubQ)
        rospy.Subscriber(stateSub, String, self.receiveState, queue_size = subQ) 

        self.batch_size = self.valueTrain['batch']
        self.state_n = self.valueParams['state_n']

        self.dataSize = 0 #number of data tuples we have accumulated so far
        self.sigmoid = nn.Sigmoid()

        self.prevState = None
        self.prevAction = None 

        self.goalPosition = 0 
        self.startDistance = []
        self.fail = False
        self.ropes = [0,1,2]

        self.valueLoss = []
        self.stop = False
        self.avgLoss = 0
        self.trainIt = 0
    
    def receiveDone(self, message):
        if message.data  == 1:
            self.stop = True
           
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

    def rewardFunction(self, state, action): 
        state = state.ravel()
        prevState = self.prevState.ravel()
        robState = state[:self.state_n]
        position = np.array(state[3:6])
        if position[2] <= .1:
            return -20 #failure
        prevPosition = np.array(prevState[3:6])

        R_curr_agent = self.agentReward(position, prevPosition, state[6], self.startDistance[0])
        
        R_agents = 0
        for i in range(self.agents_n - 1):
            startIndex = self.own_n + 4*i
            position = state[startIndex: startIndex + 3]
            if position[2] <= .1:
                return -20
            prevPosition = np.array(prevState[startIndex: startIndex+3])
            R_agents += self.agentReward(position, prevPosition, state[startIndex + 3], self.startDistance[i + 1])

        R_rope = -10 if ((self.u_n == 3) and (np.ravel(action)[-1] in self.ropes)) else 0

        reward = R_curr_agent + self.weight_agents * R_agents
        return reward

    def agentReward(self, position, prevPosition, orientation, startDistance):
        '''Calculate distance from the goal location'''
        deltas = self.goalPosition - position
        currDistance = np.sqrt(np.sum(deltas*deltas))
        difference = (startDistance - currDistance)/startDistance
        sign = 1 if difference >= 0 else -1
        R_loc = sign*np.exp(3*sign*difference) * self.weight_loc
        
        '''Calculate velocity along direction towards goal position '''
        deltas = position - prevPosition
        vector = self.goalPosition - prevPosition
        norm = np.sqrt(np.sum(np.square(vector)))
        vector = vector / norm
        dot = np.sum(deltas * vector)
        sign = 1 if dot >= 0 else -1
        R_vel = sign * np.exp(3*sign*dot) *self.weight_vel

        '''Calculate the delta of angle towards the goal'''
        delta = np.pi - (np.abs(orientation))
        sign = 1 if delta >= 0 else -1
        R_ori = sign * np.exp(delta * sign) *self.weight_ori

        return R_loc + R_vel + R_ori

    def sample(self, mean, var):
        return np.random.normal(mean, var)
    
    def plotLoss(self, valueOnly = False, title1 = "Critic Loss over Iterations", title2 = "Actor Loss over Iterations"):
        plt.plot(range(len(self.valueLoss)), self.valueLoss)
        plt.title(title1)
        plt.show()
        if not valueOnly:
            plt.plot(range(len(self.actorLoss)), self.actorLoss)
            plt.title(title2)
            plt.show()

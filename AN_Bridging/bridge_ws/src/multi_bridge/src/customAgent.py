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

class CustomAgent(Agent):
    def __init__(self, params):
        super(CustomAgent, self).__init__(params)

        actorParams = params['actorParams']
        actorTrain = params['actorTrain']

        self.discount = actorTrain['gamma']
        self.weight_loc = actorTrain['alpha1']
        self.weight_vel = actorTrain['alpha2']
        self.weight_agents = actorTrain['lambda']
        self.horizon = actorTrain['horizon']
        self.expSize = actorTrain['buffer']
        self.exploration = actorTrain['explore']

        self.own_n = actorParams['own_n']
        self.u_n = actorParams['output_n']
        self.prob = actorParams['prob']

        self.actor = Network(actorParams, actorTrain)

        self.replayFeatures = 
        self.sigmoid = nn.Sigmoid()
        self.actorLoss = []

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
        R_rope = -2 if ((self.u_n == 3) and (action[-1] in self.ropes)) else 0

        reward = self.weight_loc * R_loc + self.weight_vel * R_vel + self.weight_agents * R_agents + R_rope
        return reward


    def train(self):
        if self.dataSize >= self.batch_size:
            choices = np.random.choice(min(self.expSize, self.dataSize), self.batch_size) 
            data = self.experience[choices]
            r = data[:, self.state_n + self.u_n: self.state_n + self.u_n + 1] #ASSUMING S, A, R, S' structure for experience 
            targets = torch.from_numpy(r) + self.discount * self.critic.predict(data[:, -self.state_n: ])
            loss = self.critic.train_cust(data[:, :self.state_n], targets)
            self.criticLoss.append(loss)

            index = np.random.randint(0, self.experience.shape[0] - self.horizon)
            data = self.experience[index: index + self.horizon]
            states = data[:,:self.state_n]
            statePrime = data[:, -self.state_n:]
            valueS = self.critic.predict(states)
            valueSPrime = self.critic.predict(statePrime)
            advantage = torch.from_numpy(data[:, self.state_n + self.u_n: self.state_n + self.u_n + 1]) + self.discount*valueSPrime + valueS
            actions = data[:, self.state_n: self.state_n + self.u_n]
            loss = self.actor.train_cust(states, actions, advantage)
            self.actorLoss.append(loss)    
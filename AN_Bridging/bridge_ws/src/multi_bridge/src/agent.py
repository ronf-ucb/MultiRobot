#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import vrep

# Collaborative agent in multi-agent framework. Initiates a class that contains:
    # actor network
    # critic network
    # actor network copy for KL divergence calculations

'''
ASSUMPTION: direction of movement is in the x-directions'''

#Supports MADDPG with TRPO inspiration and optimizations in better scaling towards larger multiagent systems and stability

class Agent():
    def __init__(self, actorParams, criticParams, atrainParams, ctrainParams, ROSparams):
        rospy.init_node('Dummy', anonymous = True)
        stateSub = ROSparams['stateSub']
        subQ = ROSparams['subQueue']
        self.actionPub = ROSparams['actionPub']
        pubQ = ROSparams['pubQueue']
        self.agents_n = ROSparams['numAgents']
        self.delta_t = ROSparams['delta_t']

        rospy.Subscriber(stateSub, String, self.receiveState, queue_size = subQ) 
        self.aPub = rospy.Publisher(self.actionPub, Vector3, queue_size = pubQ)

        self.actor = None 
        self.critic = None 
        self.prevActor = None
        self.discount = atrainParams['gamma']

        self.weight_loc = atrainParams['alpha1']
        self.weight_vel = atrainParams['alpha2']
        self.weight_agents = atrainParams['lambda']
        self.batch_size = ctrainParams['batch']
        self.horizon = atrainParams['horizon']
        self.prob = actorParams['prob']

        self.state_n = criticParams['state_n']
        self.own_n = actorParams['own_n']
        self.u_n = actorParams['output_n']
        self.sigmoid = nn.Sigmoid()
        replayFeatures = 2*self.state_n + self.u_n + 1 

        self.expSize = atrainParams['buffer']
        self.experience = np.zeros((self.expSize, replayFeatures))

        self.exploration = atrainParams['explore']
        self.dataSize = 0 #number of data tuples we have accumulated so far

        self.actor = Network(actorParams, atrainParams)
        self.critic = Network(criticParams, ctrainParams)

        self.prevState = None
        self.prevAction = None 

        self.goalPosition = 0 
        self.startDistance = 0
        while(True):
            x = 1+1
        


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
        if self.prob:
            var = (out.narrow(1, self.u_n, self.u_n)).detach().numpy().ravel()
            action = self.sample(mean, var) + np.random.normal(0, self.exploration, self.u_n)
            action += np.random.normal(0, self.exploration, self.u_n)
            msg = Vector3()
            msg.x = action[0]
            msg.y = action[1]
            if (self.u_n > 2):
                action[2] = np.round(action[2])
                msg.z = action[2]
            self.aPub.publish(msg)
        else:
            i = np.random.random()
            if i < self.exploration:
                action = np.random.randint(0, self.u_n)
            else:
                action = np.argmax(mean)
            print("ERROR: NOT IMPLEMENTED")
            assert False            
        return action

    
    def sample(self, mean, var):
        return np.random.normal(mean, var)

    
    def rewardFunction(self, state, action):
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

        reward = self.weight_loc * R_loc + self.weight_vel * R_vel + self.weight_agents * R_agents
        return reward

        
    def train(self):
        if self.dataSize >= self.horizon:
            choices = np.random.choice(min(self.expSize, self.dataSize), self.batch_size) 
            data = self.experience[choices]
            r = data[:, self.state_n + self.u_n: self.state_n + self.u_n + 1] #ASSUMING S, A, R, S' structure for experience 
            targets = torch.from_numpy(r) + self.discount * self.critic.predict(data[:, -self.state_n: ])
            self.critic.train_cust(data[:, :self.state_n], targets)

            index = np.random.randint(0, self.experience.shape[0] - self.horizon)
            data = self.experience[index: index + self.horizon]
            states = data[:,:self.state_n]
            statePrime = data[:, -self.state_n:]
            valueS = self.critic.predict(states)
            valueSPrime = self.critic.predict(statePrime)
            advantage = torch.from_numpy(data[:, self.state_n + self.u_n: self.state_n + self.u_n + 1]) + self.discount*valueSPrime + valueS
            actions = data[:, self.state_n: self.state_n + self.u_n]
            self.actor.train_cust(states, actions, advantage)
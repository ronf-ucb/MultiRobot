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

class Agent(object):
    def __init__(self, params):
        self.valueParams = params['valueParams']
        self.valueTrain = params['valueTrain']
        self.ROSParams = params['ROS']

        self.valueNet = Network(self.valueParams, self.valueTrain)

        self.agents_n = self.ROSParams['numAgents']

        self.finish_sub = rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)
        self.tankPub = rospy.Publisher(self.ROSParams['tankPub'], Vector3, queue_size = self.ROSParams['pubQueue'])
        rospy.Subscriber(self.ROSParams['tankSub'], String, self.receiveState, queue_size = self.ROSParams['subQueue']) 

        self.batch_size = self.valueTrain['batch']
        self.state_n = self.valueParams['in_n']
        self.prob = self.valueParams['prob']
        self.u_n = self.valueParams['u_n']
        self.own_n = self.valueParams['own_n']
        self.other_n = self.valueParams['other_n']

        self.dataSize = 0 #number of data tuples we have accumulated so far
        self.trainMode = self.valueParams['trainMode']

        '''TODO: change this to ordered tuple'''
        self.prevState = None
        self.prevAction = None 
        self.prevDeltas = None

        self.goalPosition = 0 
        self.startDistance = []
        self.fail = False

        self.valueLoss = []
        self.avgLoss = 0

        '''TODO: change this to ordered tuple'''
        self.tankRewards = {"velocity": [], "location": [], "relativeLocation": [], "orientation": [], "relativeOrientation":[] , "total": []}
        self.bridgeRewards = {"velocity": [], "location": [], "relativeLocation": [], "orientation": [], "relativeOrientation": [], "total":[]}
        self.tankRun = {"velocity": 0, "location": 0, "relativeLocation": 0, "orientation": 0, "relativeOrientation":0 , "total": 0}
        self.bridgeRun = {"velocity": 0, "location": 0, "relativeLocation": 0, "orientation": 0, "relativeOrientation": 0, "total":0}

        self.stop = False
        self.avgLoss = 0
        self.trainIt = 0
        self.phases = [False, False, False] #bridge, across, pull
        self.discount = self.valueTrain['gamma']
        self.weight_loc = self.valueTrain['alpha1']
        self.weight_vel = self.valueTrain['alpha2']
        self.weight_ori = self.valueTrain['alpha3']
        self.weight_agents = self.valueTrain['lambda']
    
    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.manageStatus(1)
    
    def checkPhase(self, state):
        pos0 = np.array(state[:3])
        pos1 = np.array(state[self.own_n: self.own_n + 3])
        if self.distance(pos0, self.goalPosition) <.03 or self.distance(pos1, self.goalPosition) <.03:
            return 50
        if pos0[2] <= .1 or pos1[2] <= .1:
            return -10
        if not self.phases[0] and pos1[0] > -.52: #achieved BRIDGE phase. Adjust accordingly
            self.phases[0] = True 
            return 15
        if not self.phases[1] and pos0[0] > -.4: #achieved CROSS phase. Adjust accordingly
            self.phases[1] = True 
            return 20
        if not self.phases[2] and pos1[0] > -.35: #achieved PULL phase. Adjust accordingly
            self.phases[2] = True
            return 25
        return 0
        
    def rewardFunction(self, state, action): 
        state = state.ravel()
        
        r = self.checkPhase(state)
        if r != 0:
            return r

        prevState = self.prevState.ravel()
        robState = state[:self.state_n]
        position = np.array(state[:3])
        prevPosition = np.array(prevState[:3])


        R_loc, R_vel, R_ori = self.agentReward(position, prevPosition, state[3], prevState[3], self.startDistance[0])
        R_curr_agent = R_loc + R_vel +2* R_ori
        self.tankRun["location"] += R_loc 
        self.tankRun['velocity'] += R_vel
        self.tankRun["orientation"] += 2 * R_ori 

        onePosition = position
        oneOrientation = state[3]
        
        R_agents = 0
        for i in range(self.agents_n - 1):
            startIndex = self.own_n + self.other_n*i
            position = state[startIndex: startIndex + 3]
            prevPosition = np.array(prevState[startIndex: startIndex+3])
            R_loc, R_vel, R_ori = self.agentReward(position, prevPosition, state[startIndex + 3], prevState[startIndex+3], self.startDistance[i + 1])

            R_agents = R_loc + R_vel + 2*R_ori
            self.bridgeRun["location"] += R_loc 
            self.bridgeRun['velocity'] += R_vel
            self.bridgeRun["orientation"] += 2*R_ori 

            deltaX = abs(position[0]) - abs(onePosition[0])
            deltaY = abs(position[1]) - abs(onePosition[1])
            relativeLoc = self.weight_loc * (self.prevDeltas[0] - deltaX + self.prevDeltas[1] - deltaY)
            R_agents += relativeLoc  #we don't like it when the robots are far apart
            relativeOri = 0#abs(oneOrientation - state[startIndex + 3]) * self.weight_ori
            R_agents -= relativeOri
        

        self.tankRun["relativeLocation"] -= relativeLoc 
        self.bridgeRun["relativeLocation"] -= relativeLoc 
        self.tankRun["relativeOrientation"] -= relativeOri 
        self.bridgeRun["relativeOrientation"] -= relativeOri


        R_rope = -3 if action == 1 else 0
        reward = R_curr_agent + self.weight_agents * R_agents

        self.tankRun['total'] += reward
        self.bridgeRun['total'] += reward
        return reward

    def distance(self, point1, point2):
        squareSum = sum([(point1[i] - point2[i])**2 for i in range(3)])
        return np.sqrt(squareSum)

    def agentReward(self, position, prevPosition, orientation, prevOrientation, startDistance):
        '''Calculate distance from the goal location'''
        deltas = self.goalPosition - position
        scaleX = np.exp(5 - 5*abs(deltas[0]))
        scaleY = np.exp(5*abs(deltas[1]))
        R_loc = deltas[0] * 2 * self.weight_loc

        '''Add the movement in x and subtract the movement in y'''
        deltas = position - prevPosition 
        deltaX = 5 * deltas[0] * self.weight_vel #* scaleX #positive is good 
        deltaY = abs(deltas[1]) * self.weight_vel# * scaleY
        R_vel = (deltaX - deltaY)


        '''Calculate the delta of angle towards the goal. Subtract reward '''
        delta = np.abs(prevOrientation) - np.abs(orientation)
        R_ori = delta *self.weight_ori - (np.abs(orientation) * .1)

        return (R_loc, R_vel, R_ori)

    def positiveWeightSampling(self, idx):
        rewards = self.experience[:, idx].ravel()
        sumExp = np.sum(np.exp(np.abs(5*rewards)))
        softmaxProbability = np.exp(np.abs(5*rewards))/sumExp
        return softmaxProbability.tolist()

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
    
    def plotRewards(self):
        for k in self.tankRewards.keys():
            plt.plot(range(len(self.tankRewards[k])), self.tankRewards[k], label = k)
        plt.title("Tank Rewards over Episodes")
        plt.legend()
        plt.show()
        for k in self.bridgeRewards.keys():
            plt.plot(range(len(self.bridgeRewards[k])), self.bridgeRewards[k], label = k)
        plt.title("Bridge Rewards over Episodes")
        plt.legend()
        plt.show()

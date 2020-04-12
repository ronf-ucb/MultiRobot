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
        if self.trainMode:
            self.QNetwork = self.valueNet
            self.targetNetwork = Network(self.valueParams, self.valueTrain)
        else:
            self.valueNet.load_state_dict(torch.load("/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork.txt"))
        self.expSize = self.valueTrain['buffer']
        self.exploration = self.valueTrain['explore']
        self.base = self.valueTrain['baseExplore']
        self.decay = self.valueTrain['decay']
        self.step = self.valueTrain['step']
        self.replayFeatures = self.valueTrain['replayDim']
        self.experience = np.zeros((self.expSize, self.replayFeatures))
        self.bridgePub = rospy.Publisher(self.ROSParams["bridgePub"], Vector3, queue_size = 1)
        self.replaceCounter = 0
        self.actionMap = {0: (-3,-1), 1:(-1,-3), 2:(-3,-3), 3:(1,3), 4:(3,3), 5:(3,1), 6:(0,0)}
    
        while(not self.stop):
            x = 1+1
        
        self.plotLoss(True, "Centralized Q Networks: Q Value Loss over Iterations")
        self.plotRewards()
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
            for k in self.tankRun.keys():
                self.tankRewards[k].append(self.tankRun[k])
                self.bridgeRewards[k].append(self.bridgeRun[k])
                self.tankRun[k] = 0
                self.bridgeRun[k] = 0
        else:
            self.fail = False

    def receiveState(self, message):
        floats = vrep.simxUnpackFloats(message.data)
        self.goalPosition = np.array(floats[-4:-1])
        failure = floats[-1]
        state = (np.array(floats[:self.state_n])).reshape(1,-1)

        if len(self.startDistance) == 0:
            for i in range(self.agents_n): #assuming both states have same number of variables
                pos = state[:, self.own_n*i:self.own_n*i + 3].ravel()
                self.startDistance.append(np.sqrt(np.sum(np.square(pos - self.goalPosition))))
        index, ropeAction = (self.sendAction(state))
        if type(self.prevState) == np.ndarray:
            pen = 1 if self.prevAction[-1] > 50 else 0
            r = np.array(self.rewardFunction(state, pen)).reshape(1,-1)
            print(r)
            self.store(self.prevState, self.prevAction, r, state, index)
            self.dataSize += 1
        self.prevState = state
        self.prevAction = index
        state = state.ravel()
        self.prevDeltas = [abs(state[0]) - abs(state[4]), abs(state[1]) - abs(state[5]), abs(state[3]) - abs(state[7])]
        if self.trainMode:
            self.train()
        self.manageStatus(failure)
        return 

    def store(self, s, a, r, sprime, aprime = None):
        self.experience[self.dataSize % self.expSize] = np.hstack((s, a, r, sprime))
    
    def saveModel(self):
        torch.save(self.QNetwork.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QNetwork2.txt")
        print("Network saved")

    def sendAction(self, state):
        q = self.QNetwork.predict(state)
        i = np.random.random()
        if i < self.exploration:
            index = np.random.randint(self.u_n)
        else:
            index = np.argmax(q.detach().numpy())
        tank, bridge = self.index_to_action(index)
        self.tankPub.publish(tank)
        self.bridgePub.publish(bridge)
        return np.array([index]).reshape(1,-1), np.array([tank.z]).reshape(1,-1)

    def index_to_action(self, index):
        tankmsg = Vector3()
        bridgemsg = Vector3()
        if index >= 49:
            tankmsg.x = 0
            tankmsg.y = 0
            tankmsg.z = index - 49
            bridgemsg.x = 0
            bridgemsg.y = 0
        else:
            tank = self.actionMap[index // 7]
            bridge = self.actionMap[index % 7]
            tankmsg.x = tank[0]
            tankmsg.y = tank[1]
            tankmsg.z = -1
            bridgemsg.x  = bridge[0]
            bridgemsg.y = bridge[1]
        return (tankmsg, bridgemsg)
        
    def train(self):
        if self.dataSize > self.batch_size:
            probabilities = self.positiveWeightSampling(self.state_n + 1)
            choices = np.random.choice(min(self.dataSize, self.expSize), self.batch_size, probabilities)
            data = self.experience[choices]
            states = data[:, :self.state_n]
            actions = data[:, self.state_n: self.state_n + 1]
            rewards = data[: self.state_n + 1: self.state_n + 2]
            nextStates = data[:, -self.state_n:]

            if self.replaceCounter % 200 == 0:
                self.targetNetwork.load_state_dict(self.QNetwork.state_dict())
            self.replaceCounter += 1

            #PREPROCESS AND POSTPROCESS SINCE WE ARE USING PREDICT TO SEND ACTIONS
            processedInputs = self.QNetwork.preProcessIn(states) #preprocess inputs
            qValues = self.QNetwork(torch.FloatTensor(processedInputs)) #pass in
            qValues = self.QNetwork.postProcess(qValues) #post process
            q = torch.gather(qValues, 1, torch.LongTensor(actions)) #get q values of actions
            
            processedNextStates = self.QNetwork.preProcessIn(nextStates) #preprocess
            qnext = self.targetNetwork(torch.FloatTensor(processedNextStates)).detach() #pass in
            qnext = self.targetNetwork.postProcess(qnext) #postprocess
            qtar = torch.FloatTensor(rewards) + self.discount * qnext.max(1)[0].view(self.batch_size, 1) #calculate target
    
            loss = self.QNetwork.loss_fnc(q, qtar)

            self.QNetwork.optimizer.zero_grad()
            loss.backward()
            self.QNetwork.optimizer.step()
            self.avgLoss += loss/self.batch_size
            self.trainIt += 1
            if self.trainIt % self.step == 0:
                self.exploration = (self.exploration - self.base)*self.decay + self.base
                print(" ############# NEW EPSILON: ", self.exploration, " ################")
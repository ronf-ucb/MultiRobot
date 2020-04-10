#! /usr/bin/env python

import numpy as np 
from centralQ import CentralQ
import torch

class CentralQSarsa(CentralQ):
    def __init__(self, params):
        assert params['valueTrain']['batch'] == params['valueTrain']['buffer']
        self.QWeight = params['valueTrain']['QWeight']
        super(CentralQSarsa, self).__init__(params)
    
    def saveModel(self):
        torch.save(self.QNetwork.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QSARSANetwork2.txt")
        print("Network saved")

    def store(self, s, a, r, sprime, aprime = None):
        self.experience[self.dataSize % self.expSize] = np.hstack((s, a, r, sprime, aprime))

    def train(self):
        if self.dataSize == self.batch_size:
            data = self.experience
            states = data[:, :self.state_n]
            actions = data[:, self.state_n: self.state_n + 1]
            rewards = data[: self.state_n + 1: self.state_n + 2]
            nextStates = data[:, -(self.state_n + 1):-1]
            nextActions = data[:, -1:]

            if self.replaceCounter % 20 == 0:
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
            qmax = qnext.max(1)[0].view(self.batch_size, 1)
            qnext = torch.gather(qnext, 1, torch.LongTensor(nextActions)).detach()
            qtar = torch.FloatTensor(rewards) + self.discount * ((self.QWeight * qmax) + ((1-self.QWeight) * qnext))
            loss = self.QNetwork.loss_fnc(q, qtar)

            self.QNetwork.optimizer.zero_grad()
            loss.backward()
            self.QNetwork.optimizer.step()
            self.avgLoss += loss/self.batch_size
            self.trainIt += 1
            if self.trainIt % self.step == 0:
                self.exploration = (self.exploration - self.base)*self.decay + self.base
                print("NEW EPSILON: ", self.exploration)
            self.dataSize = 0

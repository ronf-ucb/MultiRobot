#! /usr/bin/env python

import numpy as np 
from centralQ import CentralQ
import torch

class CentralQSarsa(CentralQ):
    def __init__(self, params, name, task):
        assert params['valTrain']['batch'] == params['valTrain']['buffer']
        self.QWeight = params['valTrain']['QWeight']
        super(CentralQSarsa, self).__init__(params, name, task)
    
    def saveModel(self):
        torch.save(self.valueNet.state_dict(), "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/QSARSANetwork2.txt")
        print("Network saved")

    def store(self, s, a, r, sprime, aprime = None, failure = 0):
        self.exp[self.dataSize % self.expSize] = np.hstack((s, a, r, sprime, aprime))

    def train(self):
        if self.dataSize == self.batch_size:
            data = self.exp
            states = data[:, :self.state_n]
            actions = data[:, self.state_n: self.state_n + 1]
            rewards = data[: self.state_n + 1: self.state_n + 2]
            nextStates = data[:, -(self.state_n + 1):-1]
            nextActions = data[:, -1:]

            if self.replaceCounter % 20 == 0:
                self.tarNet.load_state_dict(self.valueNet.state_dict())
            self.replaceCounter += 1


            #PREPROCESS AND POSTPROCESS SINCE WE ARE USING PREDICT TO SEND ACTIONS
            qValues = self.valueNet(torch.FloatTensor(states)) #pass in
            q = torch.gather(qValues, 1, torch.LongTensor(actions)) #get q values of actions
            
            qnext = self.tarNet(torch.FloatTensor(nextStates)).detach() #pass in
            qmax = qnext.max(1)[0].view(self.batch_size, 1)
            qnext = torch.gather(qnext, 1, torch.LongTensor(nextActions)).detach()
            qtar = torch.FloatTensor(rewards) + self.discount * ((self.QWeight * qmax) + ((1-self.QWeight) * qnext))
            loss = self.valueNet.loss_fnc(q, qtar)

            self.valueNet.optimizer.zero_grad()
            loss.backward()
            self.valueNet.optimizer.step()
            self.avgLoss += loss/self.batch_size
            self.trainIt += 1
            self.lrStep += 1
            if self.lrStep % self.step == 0:
                self.explore = (self.explore - self.base)*self.decay + self.base
                print("NEW EPSILON: ", self.explore)
            self.dataSize = 0

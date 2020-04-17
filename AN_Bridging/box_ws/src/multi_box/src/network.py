#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

class Network(nn.Module):
    def __init__(self, netParams, trainParams):
        super(Network,self).__init__()
        self.in_n = netParams['in_n']
        self.out_n = netParams['out_n']
        self.prob = netParams['prob']
        self.hidden_w = netParams['hidden']
        self.depth = netParams['depth']
        self.act = netParams['act']
        self.d = netParams['dropout']
        self.lr = trainParams['lr']
        self.pre = netParams['preprocess']
        self.l2 = trainParams['l2']
        loss = netParams['loss_fnc']
        self.sigmoid = nn.Sigmoid()

        if self.prob:
            self.out_n *= 2

        if loss == "policy_gradient":
            self.loss_fnc = None
        elif loss == "MSE":
            self.loss_fnc = nn.MSELoss()

        if self.pre:
            self.scalarInput = StandardScaler() #or any of the other scalers...look into them 
            self.scalarOutput = StandardScaler()

        self.createFeatures()

        self.optimizer =  optim.Adam(super(Network, self).parameters(), lr=self.lr, weight_decay = self.l2)
    
    def createFeatures(self):
        layers = []
        layers.append(('input_lin', nn.Linear(
                self.in_n, self.hidden_w)))       # input layer
        layers.append(('input_act', self.act))
        layers.append(('input_dropout', nn.Dropout(p = self.d)))
        for d in range(self.depth):
            layers.append(('lin_'+str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('act_'+str(d), self.act))
            layers.append(('dropout_' + str(d), nn.Dropout(p = self.d)))
        layers.append(('out_lin', nn.Linear(self.hidden_w, self.out_n)))
        self.features = nn.Sequential(OrderedDict(layers))
    
    def preProcessIn(self, inputs):
        if self.pre:
            norm = inputs * 20 
            '''FOR NOW. Works because we get more distinction between points. Will do normalization eventually using ZScale'''
            return norm
        return inputs

    def postProcess(self, outputs):
        #depeneds what we are trying to do 
        if self.pre:
            return outputs 
            '''FOR NOW'''
        return outputs

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        if self.pre:
            inputs = self.preProcessIn(inputs)
        outputs = self.features(inputs)
        if self.pre:
            outputs = self.postProcess(outputs)
        return outputs


    def train_cust(self, inputs, outputs, advantages = None):
        self.train()
        self.optimizer.zero_grad()
        lossTot = 0
        for i in range(self.epochs):
            out = self.forward(inputs)
            if self.loss_fnc != None: #MSELoss
                assert advantages == None 
                loss = self.loss_fnc(out, outputs)
            else: #policy gradient
                #each row is a sample. Outputs represent our actions!
                means = out[:, :int(self.out_n/2)]
                if self.prob:
                    std = out[:, int(self.out_n/2):]
                    outputs = torch.FloatTensor(outputs)
                    prob = torch.exp(-(1/2)*(((means - outputs) ** 2 )/  std))
                    prob = (1/((2*np.pi)**(1/2) * std) * prob)
                    loss = -torch.sum(torch.log(prob)*advantages)
                else:
                    loss = (torch.sum(means * advantages))*(1 / (means.size())[0])
            loss.backward() 
            self.optimizer.step()
            lossTot += loss
        return lossTot


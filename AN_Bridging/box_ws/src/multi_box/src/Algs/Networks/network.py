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
        self.act = netParams['act']
        self.d = netParams['dropout']
        self.lr = trainParams['lr']
        self.schedule = trainParams['lr_decay']
        self.pre = netParams['preprocess']
        self.l2 = trainParams['l2']
        loss = netParams['loss_fnc']
        self.manual = trainParams['manual']
        if self.manual:
            self.mean = trainParams['mean']
            self.variance = trainParams['variance']

        if self.prob:
            self.out_n *= 2

        if loss == "MSE":
            self.loss_fnc = nn.MSELoss()

        self.createFeatures()

        self.optimizer =  optim.Adam(super(Network, self).parameters(), lr=self.lr, weight_decay = self.l2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.schedule[1], gamma = self.schedule[0])
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('relu'))

    
    def createFeatures(self):
        layers = []
        layers.append(('input_lin', nn.Linear(
                self.in_n, self.hidden_w[0])))       # input layer
        layers.append(('input_act', self.act[0]))
        layers.append(('input_dropout', nn.Dropout(p = self.d[0])))
        for d in range(1, len(self.hidden_w)):
            layers.append(('lin_'+str(d), nn.Linear(self.hidden_w[d-1], self.hidden_w[d])))
            layers.append(('act_'+str(d), self.act[d]))
            layers.append(('dropout_' + str(d), nn.Dropout(p = self.d[d])))
        layers.append(('out_lin', nn.Linear(self.hidden_w[len(self.hidden_w) - 1], self.out_n)))
        self.features = nn.Sequential(OrderedDict(layers))
        self.features.apply(self.init_weights)
    
    def preProcessIn(self, inputs):
        if self.pre:
            if self.manual:
                norm = torch.FloatTensor((inputs - self.mean) / self.variance)
            else:
                norm = inputs * 10
            return norm
        return inputs


    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        if self.pre:
            inputs = self.preProcessIn(inputs)
        outputs = self.features(inputs)
        return outputs


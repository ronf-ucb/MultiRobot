#! /usr/bin/env python

'''WORK IN PROGRESS'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt 
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

class Network(nn.Module):
    def __init__(self, netParams, trainParams):
        super(Network,self).__init__()
        self.in_n = netParams['in_n']
        self.out_n = netParams['out_n'] * 2
        self.hidden_w = netParams['hidden']
        self.mean_w = netParams['mean_width']
        self.std_w = netParams['std_width']
        self.act = netParams['act']

        self.d = netParams['dropout']
        self.lr = trainParams['lr']
        self.pre = netParams['preprocess']
        self.l2 = trainParams['l2']
        self.clamp = trainParams['clamp']
        loss = netParams['loss_fnc']
        self.manual = trainParams['manual']

        if self.manual:
            self.mean = trainParams['mean']
            self.variance = trainParams['variance']

        if loss == "MSE":
            self.loss_fnc = nn.MSELoss()

        self.createFeatures()
        self.mean_lin = nn.Linear(self.mean_w, self.out_n)
        self.std_lin = nn.Linear(self.mean_w, self.out_n)
        nn.init.xavier_uniform_(self.mean_lin.weight, gain = nn.init.calculate_gain('relu'))
        nn.init_xavier_uniform_(self.std_lin.weight, gain = nn.init.calculate_gain('relu'))

        self.optimizer =  optim.Adam(super(Network, self).parameters(), lr=self.lr, weight_decay = self.l2)
    
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
        layers.append(('out_lin', nn.Linear(self.hidden_w[len(self.hidden_w) - 1], self.mean_w))) # THIS IS DIFFERENT
        self.features = nn.Sequential(OrderedDict(layers))
        self.features.apply(self.init_weights)
    
    def preProcessIn(self, inputs):
        if self.pre:
            if self.manual:
                norm = torch.FloatTensor(5*(inputs - self.mean) / self.variance)
            else:
                norm = inputs * 10
            return norm
        return inputs

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        if self.pre:
            inputs = self.preProcessIn(inputs)
        outputs = self.features(inputs)
        mean = F.relu(self.mean_lin(outputs))
        
        log_std = F.relu(self.std_lin(outputs))
        log_std = torch.clamp(log_std, -self.clamp, self.clamp)
        std = log_std.exp()
        normal = Normal(0,1)
        z = normal.sample()
        action = torch.tanh(mean + std*z)
        log_prob = Normal(mean, std).log_prob(mean + std*z) - torch.log(1-action.pow(2) + 1e-6)
        return action, log_prob


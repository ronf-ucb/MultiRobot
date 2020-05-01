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
from network import Network
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

class SoftNetwork(Network):
    def __init__(self, netParams, trainParams):
        self.mean_w     = netParams['mean_width']
        self.std_w      = netParams['std_width']
        self.mean_range = netParams['mean_range']
        self.clamp = trainParams['clamp']
        super(SoftNetwork,self).__init__(netParams, trainParams)
        self.mean_lin   = nn.Linear(self.mean_w, int(self.out_n/2))
        self.std_lin    = nn.Linear(self.mean_w, int(self.out_n/2))
        self.init_weights(self.mean_lin)
        self.init_weights(self.std_lin)
    
    def createFeatures(self):
        layers = []
        layers.append(('input_lin', nn.Linear(
                self.in_n, self.hidden_w[0])))       # input layer
        layers.append(('input_act', self.act[0]))
        layers.append(('input_dropout', nn.Dropout(p = self.d[0])))
        layers.append(('norm_in', nn.LayerNorm(self.hidden_w[0])))
        for d in range(1, len(self.hidden_w)):
            layers.append(('lin_'+str(d), nn.Linear(self.hidden_w[d-1], self.hidden_w[d])))
            layers.append(('act_'+str(d), self.act[d]))
            layers.append(('dropout_' + str(d), nn.Dropout(p = self.d[d])))
            layers.append(('norm_'+str(d), nn.LayerNorm(self.hidden_w[d])))
        layers.append(('out_lin', nn.Linear(self.hidden_w[len(self.hidden_w) - 1], self.mean_w))) # THIS IS DIFFERENT
        layers.append(('out_norm', nn.LayerNorm(self.mean_w)))
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
        mean = self.mean_lin(outputs)
        log_std = self.std_lin(outputs)

        log_std = torch.clamp(log_std, self.clamp[0], self.clamp[1])
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.reshape(-1, int(self.out_n/2))
        log_prob = log_prob.sum(1, keepdim = True)

        action = self.mean_range * action
        return action, log_prob
    


#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torch.optim as optim
from collections import OrderedDict
from network import Network

class TD3Network(Network):
    def __init__(self, netParams, trainParams):
        super(TD3Network,self).__init__(netParams, trainParams)
        self.mean_range = netParams['mean_range']
        self.tan = nn.Tanh()

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        if self.pre:
            inputs = self.preProcessIn(inputs)
        
        outputs = self.features(inputs)
        outputs = self.mean_range * self.tan(outputs/5).view(-1, self.out_n)
        return outputs


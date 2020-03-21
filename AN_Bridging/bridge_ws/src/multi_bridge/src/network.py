#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

'''TODO: FINISH LOOKING AT ALL OF THIS'''

class Network(nn.Module):
    def __init__(self, netParams, trainParams):
        super(Network,self).__init__()
        self.state_n = netParams['state_n']
        self.in_n = self.state_n
        self.out_n = netParams['output_n']
        self.prob = netParams['prob'] #denotes whether or not this is a PNN
        self.noise_in = netParams['sigma'] 
        self.hidden_w = netParams['hidden']
        self.depth = netParams['depth']
        self.activation = netParams['activation']
        self.d = netParams['dropout']
        self.lr = trainParams['lr']
        self.pre = netParams['preprocess']
        self.post = netParams['postprocess']
        self.epochs = netParams['epochs']
        loss = netParams['loss_fnc']
        self.sigmoid = nn.Sigmoid()

        if self.prob:
            self.out_n *= 2

        assert loss == "policy_gradient" or loss == "MSE"

        if loss == "policy_gradient":
            self.loss_fnc = None
        elif loss == "MSE":
            self.loss_fnc = nn.MSELoss()

        self.scalarInput = StandardScaler() #or any of the other scalers...look into them 
        self.scalarOutput = StandardScaler()

        layers = []
        layers.append(('dynm_input_lin', nn.Linear(
                self.in_n, self.hidden_w)))       # input layer
        layers.append(('dynm_input_act', self.activation))
        layers.append(('dynm_input_dropout', nn.Dropout(p = self.d)))
        for d in range(self.depth):
            layers.append(('dynm_lin_'+str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_'+str(d), self.activation))
            layers.append(('dynm_dropout_' + str(d), nn.Dropout(p = self.d)))


        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.out_n)))
        self.features = nn.Sequential(OrderedDict(layers))

        self.optimizer =  optim.Adam(super(Network, self).parameters(), lr=self.lr)

    
    def preProcess(self, inputs, outputs):
        #normalize the input vectors
        norm = self.preProcessIn(inputs)
        self.scalarOutput.fit(outputs)
        normOut = self.scalarOutput.transform(outputs)
        return norm, normOut
    
    def preProcessIn(self, inputs):
        self.scalarInput.fit(inputs)
        norm = self.scalarInput.transform(inputs)
        return norm

    def postProcess(self, outputs):
        #depeneds what we are trying to do 
        '''TODO: Finish this'''
        return outputs
    
    def forward(self, inputs):
        x = self.features(inputs)
        x = self.postProcess(x)
        return x 
    

    def predict(self, input):
        if self.pre:
            input = self.preProcessIn(input)
        input = torch.FloatTensor(input)
        x = self.features(input)
        if self.prob:
            mean = x.narrow(1, 0, self.out_n/2)
            var = self.sigmoid(x.narrow(1, self.out_n/2, self.out_n/2))
            x = torch.cat((mean, var), dim = 1)
        return x


    def train_cust(self, inputs, outputs, advantages = None):
        self.train()
        for i in range(self.epochs):
            if self.pre:
                inputs, outputs = self.preProcess(inputs ,outputs)
            out = self.predict(inputs)
            if self.loss_fnc != None: #MSELoss
                assert advantages == None 
                loss = self.loss_fnc(out, outputs)
                loss.backward()
                self.optimizer.step()
            else: #policy gradient
                #each row is a sample. Outputs represent our actions!
                means = out[:, :int(self.out_n/2)]
                std = out[:, int(self.out_n/2):]
                outputs = torch.FloatTensor(outputs)
                prob = torch.exp(-(1/2)*(((means - outputs) ** 2 )/  std))
                prob = (1/((2*np.pi)**(1/2) * std) * prob)
                gradient = -torch.sum(torch.log(prob)*advantages)
                gradient.backward() 
                self.optimizer.step()

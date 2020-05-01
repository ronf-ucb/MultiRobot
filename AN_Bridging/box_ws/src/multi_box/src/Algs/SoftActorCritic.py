#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt
import torch.optim as optim

from Networks.network import Network
from Networks.softNetwork import SoftNetwork
from agent import Agent
from utils import positiveWeightSampling as priority
from utils import OUNoise, plot_grad_flow
from Replay import Replay

'''Twin-delayed DDPG to curb Q value overestimation with clipped double Q-learning, Q value smoothing using noise and delayed policy updates for stability'''

class SAC(Agent):
    def __init__(self, params, name, task):
        super(SAC, self).__init__(params, name, task)
        self.aPars      = params['actPars']
        self.aTrain     = params['actTrain']
        self.qPars      = params['qPars']
        self.qTrain     = params['qTrain']
        if self.trainMode:
            self.Qvals      = [Network(self.qPars, self.qTrain), Network(self.qPars, self.qTrain)]
            self.Qtars      = [Network(self.qPars, self.qTrain), Network(self.qPars, self.qTrain)]
            self.policyNet  = SoftNetwork(self.aPars, self.aTrain)
            if self.load:
                path = "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/SAC_goal1_"
                self.policyNet.load_state_dict(torch.load(path + "policy.txt"))
                self.Qvals[0].load_state_dict(torch.load(path + "Qvalue1.txt"))
                self.Qvals[1].load_state_dict(torch.load(path + "Qvalue2.txt"))
        else:
            print('Not implemented')

        for i in range(len(self.Qvals)):
            for target_param, param in zip(self.Qtars[i].parameters(), self.Qvals[i].parameters()):
                target_param.data.copy_(param)

        self.expSize    = self.vTrain['buffer']
        self.exp        = Replay(self.expSize)
        self.tau        = self.vPars['tau']
        self.out_n      = self.aPars['out_n']
        self.nu         = self.vTrain['nu']
        self.delayQ     = self.qTrain['delay']
        self.gradSteps  = self.vTrain['grad_steps']
        self.valueLoss  = []
        self.actorLoss  = []
        self.avgLoss    = 0
        self.avgActLoss = 0

        self.alpha      = self.aTrain['alpha']
        self.tar_ent    = -self.out_n
        self.log_alpha  = torch.log(torch.FloatTensor([self.alpha]))
        self.log_alpha.requires_grad = True
        self.alpha_optim = optim.Adam([self.log_alpha], lr = 3e-4)

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()

    def load_nets(self):
        pass
    
    def saveModel(self):
        path = "/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/SAC_goal2_"
        torch.save(self.policyNet.state_dict(), path + "policy.txt")
        torch.save(self.Qvals[0].state_dict(), path  + "Qvalue1.txt")
        torch.save(self.Qvals[1].state_dict(), path + "Qvalue2.txt")
        print("Network saved")
        pass
        
    def train(self):
        if self.dataSize > 1500 and self.trainMode: 
            #Unpack
            s_t, a_t, r_t, n_st, n_at, done_t = self.exp.get_data()

            for i in range(self.gradSteps): 
                #ERE Sampling
                numPoints = min(self.dataSize, self.expSize)
                sample = int(max(numPoints * (self.nu ** ((self.trainIt*1000)/15)), min(32, numPoints))) #assume episode -> 15 steps
                c = np.random.choice(sample, self.batch_size) + (numPoints - sample)

                s = torch.FloatTensor(s_t[c])
                a = torch.FloatTensor(a_t[c])
                r = torch.FloatTensor(r_t[c])
                n_s = torch.FloatTensor(n_st[c])
                done = torch.FloatTensor(done_t[c]) + 1

                #iteration updates
                self.trainIt += 1
                self.totalSteps += 1

                next_a, next_log_prob = self.policyNet(n_s)
                next_log_prob = next_log_prob.view(self.batch_size, -1)
            
                sa_next = torch.cat((n_s, next_a), dim = 1)
                sa = torch.cat((s, a), dim = 1)
                qtar = torch.min(self.Qtars[0](sa_next), self.Qtars[1](sa_next)) - self.alpha * next_log_prob 
                qtar = r + (1-done)*self.discount*qtar

                q1 = self.Qvals[0](sa)
                q2 = self.Qvals[1](sa)
                loss1 = self.Qvals[0].loss_fnc(q1, qtar.detach())
                loss2 = self.Qvals[1].loss_fnc(q2, qtar.detach())

                self.Qvals[0].zero_grad()
                loss1.backward()
                self.Qvals[0].optimizer.step()
                self.Qvals[0].scheduler.step()

                self.Qvals[1].zero_grad()
                loss2.backward()
                self.Qvals[1].optimizer.step()
                self.Qvals[1].scheduler.step()

                self.avgLoss += (loss1 + loss2) / 2

                #Policy 
                if self.totalSteps % 2 == 0:
                    new_a, log_prob = self.policyNet(s)
                    sa = torch.cat((s, new_a), dim = 1)
                    minq = torch.min(self.Qvals[0](sa), self.Qvals[1](sa))
                    loss = (self.alpha * log_prob - minq).mean()
                    self.policyNet.optimizer.zero_grad()
                    loss.backward()
                    self.policyNet.optimizer.step()
                    self.policyNet.scheduler.step()
                    self.avgActLoss += loss 
                    
                    print('Q1Loss:', loss1.detach(), '  Q2Loss: ', loss2.detach(), '  PolicyLoss:  ', loss.detach(), '    MinQAvg: ', minq.mean().detach(), '    Alpha: ', self.alpha)


                #Temperature
                new_a, log_prob = self.policyNet(s)
                alpha_loss = self.log_alpha + (-log_prob.detach() - self.tar_ent).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().detach()

                #print('Q1Loss:', loss1.detach(), '  Q2Loss: ', loss2.detach(), '  PolicyLoss:  ', loss.detach(), '    MinQAvg: ', minq.mean().detach(), '    Alpha: ', self.alpha)


                if self.totalSteps % self.delayQ == 0:
                    for i in range(len(self.Qvals)):
                        for target_param, param in zip(self.Qtars[i].parameters(), self.Qvals[i].parameters()):
                            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            
        
            

#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import math 
import rospy
import torch.nn.functional as F
from torch.distributions import Categorical
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from agent import Agent
from Networks.network import Network
from Networks.autoFeudalNetwork import AutoFeudalNet
from Buffers.FeudalBuffer import Memory
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Temp = namedtuple('Temp',('goal', 'policy', 'm_value', 'w_values_ext', 'w_values_int','m_state'))

class AutoFeudal(object):
    def __init__(self, params, name, task):
        self.task           = task
        self.vTrain         = params['train']
        self.agents         = params['agents']
        self.fun            = params['fun']
        self.name           = name

        self.actionMap      = {0: (-2,-1), 1:(-1,-2), 2:(-2,-2), 3:(1,2), 4:(2,2), 5:(2,1), 6: (-2, 2), 7: (2, -2)}

        self.pubs = {}
        for key in self.agents.keys():
            bot             = self.agents[key]
            self.pubs[key]  = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)
        rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)

        self.actions        = self.fun['u']
        self.horizon        = self.fun['c']
        self.k              = self.fun['k']
        self.d              = self.fun['d']
        self.s_reduce       = self.fun['s_reduce']
        self.num_agents     = self.fun['num_agents']
        self.valueLoss      = []

        self.net            = AutoFeudalNet(self.fun)#.to(device)

        self.m_discount     = self.vTrain['m_gamma']
        self.w_discount     = self.vTrain['w_gamma']
        self.lr             = self.vTrain['lr']
        self.trainMode      = self.vTrain['trainMode']
        self.clip_grad_norm = self.vTrain['clip_grad']
        self.step           = self.vTrain['step']
        self.alpha          = self.vTrain['alpha']
        self.stop           = False

        self.exp            = [Memory() for i in range(self.num_agents)]
        self.optimizer      = optim.Adam(self.net.parameters(), lr = self.lr)
        self.temp           = None
        self.totalSteps     = 0

        self.reset()

        task.initAgent(self)

        while(not self.stop):
            x = 1+1

        task.postTraining()

    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.task.restartProtocol(restart = 1)

    def get_action(self, s, w_s):
        net_out = self.net(torch.FloatTensor(s), w_s, self.m_lstm, self.w_lstm, self.goals_horizon)
        policies, goals, self.goals_horizons, self.m_lstm, self.w_lstm, m_value, w_values_ext, w_values_int, m_state = net_out
        self.temp = Temp(goals, policies, m_value, w_values_ext, w_values_int, m_state)
        actions, choices = ([], [])
        for p in policies:
            choice = np.asscalar(self.choose(p))
            choices.append(choice)
            action = np.array(self.actionMap[choice]).ravel()
            actions.append(action)
        return actions, choices

    def choose(self, policies):
        m = Categorical(policies)
        action = m.sample()
        action = action.data.cpu().numpy()
        return action
    
    def saveModel(self):
        pass

    def store(self, s, a, r, sprime, aprime, done):
        for i in range(self.num_agents):
            self.exp[i].push(s, a[i], r[i], 1 - done, self.temp.goal[i], self.temp.policy[i], self.temp.m_value, 
                        self.temp.w_values_ext[i], self.temp.w_values_int[i], self.temp.m_state)
        

    def reset(self):
        m_hx                = torch.zeros(1, self.d).to(device)
        m_cx                = torch.zeros(1, self.d).to(device)
        self.m_lstm         = (m_hx, m_cx)
        self.w_lstm, self.goals_horizon = ([],[])
        for i in range(self.num_agents):

            w_hx                = torch.zeros(1, self.actions * self.k).to(device)
            w_cx                = torch.zeros(1, self.actions * self.k).to(device)
            self.w_lstm.append((w_hx, w_cx))

            horizon = torch.zeros(1, self.horizon + 1, self.d).to(device)
            self.goals_horizon.append(horizon)
        self.temp = None
        return 

    def get_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** (1. / 2)
        return grad_norm

    def get_returns(self, rewards, masks, gamma, values):
        returns = torch.zeros_like(rewards)
        running_returns = values[-1].squeeze()

        for t in reversed(range(0, len(rewards)-1)):
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            returns[t] = running_returns

        if returns.std() != 0:
            returns = (returns - returns.mean()) / returns.std()

        return returns

    def get_transition_info(self, transition):
        states = torch.Tensor(transition.state).to(device)
        actions = torch.Tensor(transition.action).long().to(device)
        rewards = torch.Tensor(transition.reward).to(device)
        masks = torch.Tensor(transition.mask).to(device)
        goals = torch.stack(transition.goal).to(device)
        policies = torch.stack(transition.policy).to(device)
        m_states = torch.stack(transition.m_state).to(device)
        m_values = torch.stack(transition.m_value).to(device)
        w_values_ext = torch.stack(transition.w_value_ext).to(device)
        w_values_int = torch.stack(transition.w_value_int).to(device)
        return states, actions, rewards, masks, goals, policies, m_states, m_values, w_values_ext, w_values_int 


    def train(self):
        if len(self.exp[0]) > self.step:
            #Only get rewards for now
            transition = self.exp[0].sample()
            rewards1 = torch.Tensor(transition.reward).to(device)
            transition = self.exp[1].sample()
            rewards2 = torch.Tensor(transition.reward).to(device)
            masks = torch.Tensor(transition.mask).to(device)
            m_values = torch.stack(transition.m_value).to(device)

            #Calculate parent/manager returns 
            totRewards = rewards1 + rewards2
            m_returns = self.get_returns(totRewards, masks, self.m_discount, m_values)
            loss = 0

            for k in range(self.num_agents):
                # Unpack replay for child k
                transition = self.exp[k].sample()
                states, actions, rewards, masks, goals, policies, m_states, m_values, w_values_ext, w_values_int = self.get_transition_info(transition)

                # Child returns
                w_returns = self.get_returns(rewards, masks, self.w_discount, w_values_ext)
                intrinsic_rewards = torch.zeros_like(rewards).to(device)
                
                # Child intrinsic reward for following goal
                for i in range(self.horizon, len(rewards)):
                    cos_sum = 0
                    for j in range(1, self.horizon + 1):
                        alpha = m_states[i] - m_states[i - j]
                        beta = goals[i - j]
                        cosine_sim = F.cosine_similarity(alpha, beta)
                        cos_sum = cos_sum + cosine_sim
                    intrinsic_reward = cos_sum / self.horizon
                    intrinsic_rewards[i] = intrinsic_reward.detach()
                returns_int = self.get_returns(intrinsic_rewards, masks, self.w_discount, w_values_int)
                w_loss = torch.zeros_like(m_returns).to(device)


                # Worker actor/policy loss 
                for i in range(0, len(rewards)-self.horizon):
                    log_policy = torch.log(policies[i] + 1e-5)
                    w_advantage = w_returns[i] + returns_int[i] - w_values_ext[i].squeeze(-1) - w_values_int[i].squeeze(-1)
                    log_policy = log_policy.gather(-1, actions[i].unsqueeze(-1).unsqueeze(-1))
                    w_loss[i] = - (w_advantage) * log_policy.squeeze(-1)
                
                # Value losses and add to loss
                w_loss = w_loss.mean()
                w_loss_value_ext = F.mse_loss(w_values_ext.view(w_returns.size()), w_returns.detach())
                w_loss_value_int = F.mse_loss(w_values_int.view(w_returns.size()), returns_int.detach())
                loss += w_loss + w_loss_value_ext + w_loss+ w_loss_value_int

            # Calculate parent/manager loss 
            m_loss = torch.zeros_like(w_returns).to(device)
            for i in range(0, len(rewards)-self.horizon):
                m_advantage = m_returns[i] - m_values[i].squeeze(-1)
                alpha = m_states[i + self.horizon] - m_states[i]
                beta = goals[i]
                cosine_sim = F.cosine_similarity(alpha.detach(), beta)
                m_loss[i] = - m_advantage * cosine_sim
            m_loss = m_loss.mean()
            m_loss_value = F.mse_loss(m_values.view(m_returns.size()), m_returns.detach())
            loss = m_loss + m_loss_value 


            # Optimizer step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
            self.optimizer.step()


            # Detach hidden states and reset memory
            m_hx, m_cx = self.m_lstm
            self.m_lstm = (m_hx.detach(), m_cx.detach())
            for i in range(self.num_agents):
                w_hx, w_cx = self.w_lstm[i]
                self.w_lstm[i] = (w_hx.detach(), w_cx.detach())
                self.goals_horizon[i] = self.goals_horizon[i].detach()
                self.exp[i] = Memory()

            # Increments
            self.totalSteps += 1
            self.valueLoss.append(loss)
            print('Value/Actor Loss: ', loss)
            print('')

            return loss


    

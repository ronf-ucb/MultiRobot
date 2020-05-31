#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import math 
import rospy
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from agent import Agent
from Networks.network import Network
from Networks.feudalNetwork import FeudalNetwork
from Buffers.CounterFeudalBuffer import WorkerMemory, ManagerMemory
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_Temp = namedtuple('Temp',('goal', 'state'))
w_Temp = namedtuple('Temp', ('policies', 'goal'))


class CounterFeudal(object):
    def __init__(self, params, name, task):
        self.name           = name
        self.task           = task
        self.vTrain         = params['valTrain'] # Counterfactual network
        self.vPars          = params['valPars']
        self.aTrain         = params['actTrain'] # Local Actors
        self.aPars          = params['actPars']
        self.m_params       = params['m_pars'] # Manager
        self.m_train        = params['m_train']
        self.agents         = params['agents'] # Agents

        self.pubs = {}
        self.actionMap        = {0: (-2,-1), 1:(-1,-2), 2:(-2,-2), 3:(1,2), 4:(2,2), 5:(2,1), 6: (-2, 2), 7: (2, -2)} 
        for key in self.agents.keys():
            bot             = self.agents[key]
            self.pubs[key]  = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)
        rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)

        self.tau            = self.vPars['tau']
        self.int_weight     = self.vPars['int_weight']
        self.trainMode      = self.vPars['trainMode']
        self.step           = self.vTrain['step']
        self.td_lambda      = .8

        self.h_state_n      = self.aPars['u_n'] * self.aPars['k']
        self.c              = self.m_params['c']
        self.w_discount     = self.vTrain['gamma']
        self.m_discount     = self.m_train['gamma']
        self.clip_grad_norm = self.aTrain['clip']
        self.prevState      = None

        self.manager_exp    = ManagerMemory()
        self.worker_exp     = WorkerMemory()
        self.valueLoss      = []
        self.actorLoss      = []
        self.temp_first, self.temp_second= (None, None)
        self.temp_manager   = None 
        self.iteration      = 0
        self.totalSteps     = 0
        self.reward_manager = 0
        
        self.phis = [nn.Linear(self.m_params['x_state_n'], self.aPars['k'], bias=False) for i in range(len(self.agents))] # add linear layers for each actor 
        self.manager = CounterManager(self.m_params, self.m_train) # manager w/ value embedded
        self.actor = CounterActor(self.aPars, self.aTrain) # actor

        self.m_critic = Network(self.m_params, self.m_train)
        self.m_critic_target = Network(self.m_params, self.m_train)

        self.critic = Network(self.vPars, self.vTrain) # counterfactual
        self.target = Network(self.vPars, self.vTrain)

        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.m_critic_target.parameters(), self.m_critic.parameters()):
            target_param.data.copy_(param.data)

        self.reset()

        task.initAgent(self)
        self.stop = False
        while(not self.stop):
            x = 1+1

        task.postTraining()

    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.task.restartProtocol(restart = 1)

    def get_action(self, s_true, s_split):
        if self.iteration % self.c == 0: 
            self.goal = self.manager(torch.FloatTensor(s_true)) # get a new goal
            self.temp_manager = m_Temp(self.goal, torch.FloatTensor(s_true))
        else: # Use goal transition 
            self.goal = self.prevState + self.goal - torch.FloatTensor(s_true)

        embed = [phi(self.goal.detach()) for phi in self.phis]
        policies_hidden = [self.actor(torch.FloatTensor(s_split[i]), embed[i], self.h[i], i) for i in range(len(s_split))]
        policies = [r[0] for r in policies_hidden]
        self.h = [r[1] for r in policies_hidden]
        
        act_indices = [self.choose(policy) for policy in policies]

        self.temp_second = self.temp_first
        self.temp_first = w_Temp(policies, self.goal.detach())
        self.prevState = torch.FloatTensor(s_true)

        actions = [self.actionMap[index] for index in act_indices]
        self.iteration += 1

        return np.array(actions), act_indices

    def choose(self, policies):
        m = Categorical(policies)
        action = m.sample()
        action = action.data.cpu().numpy()
        if action.size == 1:
            return np.asscalar(action)
        return torch.Tensor(action).unsqueeze(1)
    
    def saveModel(self):      
        pass

    def store(self, s, a, r, sprime, aprime, done, s_w):
        if self.iteration % self.c == 1 and self.iteration != 1: # remember, we push when we're one step 
            self.manager_exp.push(self.temp_manager.state.detach().numpy(), self.goal, self.reward_manager, s, 1-done) # store into manager exp
            self.reward_manager = 0
        self.reward_manager += r
        self.worker_exp.push(s, self.temp_second.policies, self.temp_second.goal, a, r, 1 - done, aprime, sprime)

    def reset(self):
        self.iteration = 0
        self.h = [torch.zeros(1, 1, self.h_state_n).to(device) for i in range(len(self.agents))]
        self.temp_first, self.temp_second = (None, None)
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

    def zipStack(self, data):
        data        = zip(*data)
        data        = [torch.stack(d).squeeze().to(device) for d in data]
        return data

    def get_lambda_targets(self, rewards, mask, gamma, target_qs):
        target_qs = target_qs.squeeze()
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[-1] = target_qs[-1] * mask[-1]

        for t in range(ret.size()[0] - 2, -1,  -1): #TODO: check that the mask is applied correctly
            ret[t] = self.td_lambda * gamma * ret[t + 1] + \
                mask[t] * (rewards[t] + (1 - self.td_lambda) * gamma * target_qs[t + 1])
        return ret.unsqueeze(1)

    def train(self, episode_done = False): 
        if len(self.worker_exp) > self.step:
            # UNPACK REPLAY 
            m_transition    = self.manager_exp.sample()
            w_transition    = self.worker_exp.sample()

            # Manager unpack
            m_states        = torch.Tensor(m_transition.state).squeeze().to(device)
            m_actions       = torch.stack(m_transition.action).squeeze().to(device)
            m_rewards       = torch.Tensor(m_transition.reward).unsqueeze(1).to(device)
            m_next_states   = torch.Tensor(m_transition.next_state).squeeze().to(device) 
            m_masks         = torch.Tensor(m_transition.mask).unsqueeze(1).to(device)

            # Worker unpack
            w_states        = torch.Tensor(w_transition.state).squeeze().to(device)
            w_actions       = torch.Tensor(w_transition.action).float().to(device)
            w_rewards       = torch.Tensor(w_transition.reward).unsqueeze(1).to(device)
            w_next_states   = torch.squeeze(torch.Tensor(w_transition.next_state)).to(device) 
            w_masks         = torch.Tensor(w_transition.mask).unsqueeze(1).to(device) 
            w_next_actions  = torch.Tensor(w_transition.next_action).float().to(device) # on_policy training
            w_policies      = self.zipStack(w_transition.policy)
            w_goals         = torch.stack(w_transition.goal).squeeze().to(device)

            # To update the manager
            v_tar = m_rewards + self.m_discount * self.m_critic_target(m_next_states.detach()) * m_masks
            v = self.m_critic(m_states.detach())
            m_critic_loss = self.m_critic.get_loss(v, v_tar.detach())

            v_tar = m_rewards + self.m_discount * self.m_critic(m_next_states) * m_masks
            v = self.m_critic(m_states)
            advantage = (v_tar - v).detach()
            deltas = m_next_states - m_states
            m_actor_loss = (-1 * advantage * F.cosine_similarity(deltas.detach(), m_actions)).mean()
            self.manager.optimizer.zero_grad()  
            self.m_critic.optimizer.zero_grad()
            loss = m_critic_loss + m_actor_loss
            loss.backward(retain_graph=True)
            self.m_critic.optimizer.step()
            self.manager.optimizer.step()

            # add in intrinsic reward for following goal to the worker rewards w_rewards
            intrinsic = -(torch.norm(w_states + w_goals - w_next_states, dim=1) * self.int_weight).detach().unsqueeze(1)
            r = w_rewards + intrinsic 

            critic_loss = 0
            for agent in range(len(self.agents)):
                #passing in true state, actions of all agents except current, agent ID
                ID = torch.Tensor(w_states.size()[0], 1).fill_(agent)
                inp = torch.cat((w_next_states, w_next_actions[:, :agent], w_next_actions[:, agent + 1:], ID), dim = 1)
                q_tar = self.target(inp).detach().gather(1, w_next_actions[:, agent].long().unsqueeze(1))
                q_tar = self.get_lambda_targets(r.squeeze(), w_masks.squeeze(), self.w_discount, q_tar)
                inp = torch.cat((w_states, w_actions[:, :agent], w_actions[:, agent + 1:], ID), dim = 1)
                q = self.critic(inp).gather(1, w_actions[:, agent].long().unsqueeze(1))
                loss = self.critic.get_loss(q, q_tar)
                critic_loss += loss
            
            actor_loss = 0
            for agent in range(len(self.agents)):
                ID = torch.Tensor(w_states.size()[0], 1).fill_(agent)
                inp = torch.cat((w_states, w_actions[:, :agent], w_actions[:, agent + 1:], ID), dim = 1)
                q_out = self.critic(inp) #batch x num_actions
                policy = w_policies[agent] #batch x num_actions
                policy = torch.transpose(policy, 0, 1) #transpose
                mat = torch.mm(q_out, policy)
                baseline = torch.diagonal(mat, 0).detach() #diagonal elements are your baselines! 
                #gather the proper q_out elements
                q_taken = q_out.gather(1, w_actions[:, agent].long().unsqueeze(1))
                coma = (q_taken - baseline).detach()
                policy = torch.transpose(policy, 0, 1) #tranpose back
                probs_taken = policy.gather(1, w_actions[:, agent].long().unsqueeze(1))
                probs_taken[w_masks == 0] = 1.0
                loss = -(torch.log(probs_taken) * coma * w_masks).sum() / w_masks.sum()
                actor_loss += loss 

            loss = actor_loss + critic_loss 
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
            self.critic.optimizer.step()
            self.actor.optimizer.step()

            self.manager_exp = ManagerMemory()
            self.worker_exp = WorkerMemory()
            self.totalSteps += 1

            #UPDATE TARGET NETWORK:
            for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.m_critic_target.parameters(), self.m_critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            return 


class CounterManager(nn.Module):
    def __init__(self, params, train):
        # define feed forward network here
        super(CounterManager, self).__init__()
        self.width      = params['width']
        self.x_state_n  = params['x_state_n']
        self.lr         = train['lr']
        self.mean       = params['mu']
        self.std        = params['std']

        self.fc1        = nn.Linear(self.x_state_n, self.width)
        self.fc2        = nn.Linear(self.width, self.width)
        self.fc3        = nn.Linear(self.width, self.x_state_n)
        self.optimizer  = optim.Adam(super(CounterManager, self).parameters(), lr=self.lr)
        return

    def preprocess(self, inputs):
        return (inputs - self.mean) / self.std
    
    def forward(self, s_true):
        inpt = self.preprocess(s_true) 
        inpt = F.leaky_relu(self.fc1(inpt))
        inpt = F.leaky_relu(self.fc2(inpt))
        out  = F.leaky_relu(self.fc3(inpt))
        return out

class CounterActor(nn.Module):
    def __init__(self, params, train):
        super(CounterActor, self).__init__()
        self.h_state_n  = params['h_state_n']
        self.x_state_n  = params['x_state_n']
        self.u_n        = params['u_n']
        self.k          = params['k']
        self.lr         = train['lr']
        self.mean       = params['mu']
        self.std        = params['std']

        self.fc1        = nn.Linear(self.x_state_n, self.h_state_n)
        self.gru        = nn.GRU(self.h_state_n, self.u_n * self.k, num_layers = 1, batch_first = False) # expected (sequence, batch, hidden_size)
        self.soft       = nn.Softmax(dim = 1)
        self.optimizer  = optim.Adam(super(CounterActor, self).parameters(), lr=self.lr)

    def preprocess(self, inputs):
        return (inputs - self.mean) / self.std

    def forward(self, x, goal_embed, h, i):
        #TODO: Implement epsilon lower-bounds

        x = self.preprocess(x)
        inp = self.fc1(x)
        inp = F.leaky_relu(inp)
        inp = torch.unsqueeze(inp, 0)
        out, h_new = self.gru(inp, h)

        embed  = out.view(self.k, self.u_n)
        goal_embed = goal_embed.unsqueeze(0)
        embed = embed.unsqueeze(0)

        policy = torch.bmm(goal_embed, embed)
        policy = policy.squeeze(0)
        policy = self.soft(policy)
        return policy , h_new
    

    
    


    

#! /usr/bin/env python

#! /usr/bin/env python

from task import distance as dist 
from task import Task, dot, unitVector
import numpy as np 
import torch 
import torch.nn as nn
import vrep
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt

class GoalSetTask(Task):
    def __init__(self, action):
        super(GoalSetTask, self).__init__(action)

        self.prev = {"S": None, "A": None}
        self.actionMap = {0: (-2,-1), 1:(-1,-2), 2:(-2,-2), 3:(1,2), 4:(2,2), 5:(2,1), 6:(0,0)}
        self.restart = rospy.Publisher('/restart', Int8, queue_size = 1)
        rospy.Subscriber('/restart', Int8, self.restartCall, queue_size = 1)

        self.currReward = 0
        self.rewards = []
        self.currIt = 0
        self.prevIt = 0
        self.goal = 0
        self.sigma = .3
        self.c = 15
        self.distances = []


    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.pubs = self.agent.pubs
        self.out_n = self.agent.out_n
        self.trainMode = self.agent.trainMode
        self.explore = self.agent.explore
        self.name = self.agent.name
 
    def sendAction(self, s):
        msg = Vector3()
        if self.a == "argmax":
            q = self.valueNet(s)
            i = np.random.random()
            if i < self.explore:
                index = np.random.randint(self.out_n)
            else:
                q = q.detach().numpy()
                index = np.argmax(q)
            msg.x, msg.y = self.actionMap[index]
            action = np.array([index])
        if self.a == "p_policy":
            output = torch.flatten(self.policyNet(torch.FloatTensor(s)))
            action_mean = output[:self.out_n]
            action_logstd = output[self.out_n:]
            action_std = torch.exp(action_logstd)
            action = torch.normal(action_mean, action_std).detach().numpy()
            msg.x, msg.y = (action[0], action[1])
        if self.a == "d_policy":
            output = self.policyNet(torch.FloatTensor(s))
            i = np.random.random()
            output = output.detach().numpy() 
            noise = self.agent.noise.get_noise(t = 1)
            output = np.clip(output + noise, -self.agent.mean_range, self.agent.mean_range)
            print(noise, '    ', output)
            rav = np.ravel(output)
            msg.x = rav[0]
            msg.y = rav[1]
            self.pubs[self.name].publish(msg)
            return output
        self.pubs[self.name].publish(msg)
        return action.reshape(1,-1)
    
    def rewardFunction(self, s_n, a):
        currDist = dist(s_n,np.zeros(s_n.shape))
        self.distances.append(currDist)
        if currDist < .25:
            return 50
        prev = self.prev['S']
        prevOri = unitVector(prev)
        ori = unitVector(s_n)
        r_ori = abs(ori[0]) - abs(prevOri[0]) 
        deltDist = 10* (dist(prev, np.zeros(prev.shape)) - dist(s_n, np.zeros(s_n.shape)))
        return deltDist + r_ori*2 - currDist/3

    def receiveState(self, msg):
        s = np.array(vrep.simxUnpackFloats(msg.data))
        finish = 0  

        self.prevIt = self.currIt 
        a = (self.sendAction(s))

        if type(self.prev["S"]) == np.ndarray:
            r = np.array(self.rewardFunction(s, self.prev['A'])).reshape(1,-1)
            if r == 50:
                finish = 1
                print('#### SUCCESS!!!! ####')
            self.agent.store(self.prev['S'].reshape(1,-1), self.prev["A"], r, s.reshape(1,-1), None, finish)
            self.agent.dataSize += 1
            self.currReward += np.asscalar(r)
            #print('Delta: ', s, '    reward:    ', r)

        self.prev["S"] = s
        self.prev["A"] = a.reshape(1,-1)
        s = s.ravel()
        if self.trainMode and self.agent.dataSize >= self.agent.batch_size:
            self.agent.train()
        self.currIt += 1
        if self.currIt > self.c or finish:
            msg = Int8()
            msg.data = 1
            self.restart.publish(msg)
            return
        
    def restartCall(self, msg):
        if msg.data == 1:
            self.restartProtocol(1)
    
    def restartProtocol(self, restart): 
        if restart == 1:
            print('Results:     Cumulative Reward: ', self.currReward, '    Steps: ', self.agent.totalSteps, '      Closest: ', min(self.distances))
            print("")
            for k in self.prev.keys():
                self.prev[k] = None
            if self.agent.trainIt > 0:
                self.agent.valueLoss.append((self.agent.avgLoss)/self.agent.trainIt)
                self.agent.avgLoss = 0 
                if self.a == 'p_policy' or self.a == 'd_policy':
                    self.agent.actorLoss.append((self.agent.avgActLoss)/self.agent.trainIt)  
                    self.agent.avgActLoss = 0 
            self.rewards.append(self.currReward/(max(self.currIt, 1)))
            self.currIt, self.goal, self.agent.trainIt, self.currReward = (0,0,0,0)
            self.prevIt = self.currIt
            self.distances = []

    ######### POST TRAINING #########
    def postTraining(self):
        valueOnly = True if self.a == "argmax" else False
        self.plotLoss(valueOnly, "Value Loss over Iterations", "Actor Loss over Iterations")
        self.plotRewards()
        self.agent.saveModel()
        print("Total steps: ", self.agent.totalSteps)
    
    def plotLoss(self, valueOnly = False, title1 = "Critic Loss over Iterations", title2 = "Actor Loss over Iterations"):
        plt.plot(range(len(self.agent.valueLoss)), self.agent.valueLoss)
        plt.title(title1)
        plt.show()
        if not valueOnly:
            plt.plot(range(len(self.agent.actorLoss)), self.agent.actorLoss)
            plt.title(title2)
            plt.show()
    
    def plotRewards(self):
        x = range(len(self.rewards))
        plt.plot(x, self.rewards)
        plt.title("Average rewards over Episodes")
        plt.legend()
        window= np.ones(int(5))/float(5)
        lineRewards = np.convolve(self.rewards, window, 'same')
        plt.plot(x, lineRewards, 'r')
        grid = True
        plt.show()
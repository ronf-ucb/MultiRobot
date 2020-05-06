#! /usr/bin/env python

from task import Task, unitVector, dot, vector
from task import distance as dist
import numpy as np 
import rospy
import torch 
import torch.nn as nn
import vrep
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt

class BoxTask(Task):
    def __init__(self):
        super(BoxTask, self).__init__()

        self.prev = {"S": None, "A": None}
        self.s_n = 12

        self.currReward = 0
        self.rewards = []
        self.phase = 1

    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.pubs = self.agent.pubs
        self.actions = self.agent.actions
        self.trainMode = self.agent.trainMode
        self.name = self.agent.name
        self.w_phase1 = self.vTrain['w_phase1']
        self.w_phase2 = self.vTrain['w_phase2']
        self.w_phase3 = self.vTrain['w_phase3']
        rospy.Subscriber(self.agents[self.name]['sub'], String, self.receiveState, queue_size = 1) 
 
    def sendAction(self, s):
        msg = Vector3()
        action, ret = self.agent.get_action(s)
        msg.x, msg.y = (action[0], action[1])
        self.pubs[self.name].publish(msg)
        return ret

    def rewardFunction(self, s, a):
        s = s.ravel()
        prevS = self.prev["S"].ravel()
        prevPos = np.array(prevS[:3])
        pos = np.array(s[:3])
        blockPos = np.array(s[6:9])
        prevBlock = np.array(prevS[6:9])
        ori = s[5]
        prevOri = prevS[5]
        blockOri = s[11]
        if pos[-1] < .35:
            return (-3, 1)
        #reg = 0#.01 * np.sum(2 - np.abs(a)) if self.a != "argmax" else 0

        if self.phase == 1:
            if abs(pos[0] - blockPos[0]) < .7 and abs(pos[1] - blockPos[1]) < .5 and abs(ori - blockOri) < .3:
                print("### Phase ", self.phase, " completed")
                self.phase += 1
                return (5, 0)
            dist_r = (dist(prevPos, blockPos) - dist(pos, blockPos))
    
            prevVec = unitVector(vector(prevOri))
            vec = unitVector(vector(ori))
            goal = unitVector(blockPos[:2]-pos[:2])

            prevDot = dot(prevVec, goal)
            currDot = dot(vec, goal)
            ori_r = currDot - prevDot

            return ((dist_r+ 2*ori_r) * self.w_phase1, 0)


        if self.phase == 2:
            if blockPos[2] < .3:
                print("### Phase ", self.phase, " completed")
                self.phase+=1
                return (5, 0)
            block_r = blockPos[0] - prevBlock[0]
            rob_r = pos[0] - prevPos[0]
            dist_r = .5*(dist(prevPos, blockPos) - dist(pos, blockPos))
            #deltaOri = (np.abs(prevS[11]) - np.abs(blockOri)) + (np.abs(prevS[5]) - np.abs(ori))

            return ((block_r + rob_r + dist_r) * self.w_phase1, 0)

        if self.phase == 3:
            if pos[0] > .75:
                print("### Success!")
                return (5, 1)
            goal = np.array([.80, blockPos[1], pos[2]])
            delta = dist(pos, goal)
            prevDelta = dist(prevPos, goal)
            y_r = -abs(blockPos[1]-pos[1])
            return ((prevDelta - delta + .15*y_r - .05 * abs(ori)) * self.w_phase3, 0)
    
    def receiveState(self, msg):
        floats = vrep.simxUnpackFloats(msg.data)
        self.goal = np.array(floats[-4:-1])
        fail = floats[-1]
        restart = 0
        floats = floats[:self.s_n]

        if self.phase == 1:
            floats.append(0)
        else:
            floats.append(1)

        s = (np.array(floats)).reshape(1,-1)
        a = (self.sendAction(s))

        if type(self.prev["S"]) == np.ndarray:
            r, restart = self.rewardFunction(s,a)
            self.agent.store(self.prev['S'], self.prev["A"], np.array([r]).reshape(1, -1), s, a, restart)
            self.currReward += r

        if self.trainMode:
            loss = self.agent.train()

        self.prev["S"] = s
        self.prev["A"] = a
        self.restartProtocol(restart or fail)

        return 
    
    def restartProtocol(self, restart):
        if restart == 1:
            print('Results:     Cumulative Reward: ', self.currReward, '    Steps: ', self.agent.totalSteps)
            print("")
            for k in self.prev.keys():
                self.prev[k] = None
            self.goal = 0
            if self.currReward != 0:
                self.rewards.append(self.currReward)
            self.currReward = 0
            self.phase = 1
            self.agent.reset()

    ######### POST TRAINING #########
    def postTraining(self):
        #valueOnly = True if self.a == "argmax" else False
        self.plotRewards()
        self.plotLoss(True, 'Loss Over Iterations w/ Moving Average', "Actor Loss over Iterations w/ Moving Average")
        #self.agent.saveModel()
    
    def plotLoss(self, valueOnly = False, title1 = "Critic Loss over Iterations", title2 = "Actor Loss over Iterations"):
        x = range(len(self.agent.valueLoss))
        plt.plot(x, self.agent.valueLoss)
        plt.title(title1)
        plt.legend()
        window= np.ones(int(15))/float(15)
        line = np.convolve(self.agent.valueLoss, window, 'same')
        plt.plot(x, line, 'r')
        grid = True
        plt.show()
        if not valueOnly:
            plt.plot(range(len(self.agent.actorLoss)), self.agent.actorLoss)
            plt.title(title2)
            plt.show()
    
    def plotRewards(self):
        x = range(len(self.rewards))
        plt.plot(x, self.rewards)
        plt.title("Rewards Over Episodes w/ Moving Average")
        plt.legend()
        window= np.ones(int(15))/float(15)
        lineRewards = np.convolve(self.rewards, window, 'same')
        plt.plot(x, lineRewards, 'r')
        grid = True
        plt.show()
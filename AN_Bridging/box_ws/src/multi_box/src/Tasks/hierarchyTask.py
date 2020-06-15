#! /usr/bin/env python

from task import Task, unitVector, dot, vector
from task import distance as dist
import numpy as np 
import math
import rospy
import torch 
import torch.nn as nn
import vrep
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt

class HierarchyTask(Task):
    def __init__(self):
        super(HierarchyTask, self).__init__()
        self.prev = {"S": None, "A": None}
        self.actionMap = {0: "MOVE_TOWARDS", 1: "ANGLE_TOWARDS", 2:"MOVE_AWAY", 3: "FORWARD", 4: "CIRCLE_C", 5: "CIRCLE_CC", 6: "ANGLE_SAME"} 
        self.travel_gain = 1
        self.circle_gain = 3
        self.rotate_gain = 10
        self.s_n = 12
        self.currReward = 0
        self.rewards = []
        self.phase = 1
        self.counter = 0
        self.period = 50

    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.pubs = self.agent.pubs
        self.trainMode = self.agent.trainMode
        self.name = self.agent.name
        rospy.Subscriber(self.agents[self.name]['sub'], String, self.receiveState, queue_size = 1) 
 
    def sendAction(self, s, changeAction=True):
        msg = Vector3()
        if changeAction:
            ret = self.agent.get_action(s)
            self.radius = np.linalg.norm(self.goal)
            print(self.actionMap[ret])
        else:
            ret = self.prev['A']
        action = self.getPrimitive(s, self.actionMap[ret])
        msg.x, msg.y = (action[0], action[1])
        self.pubs[self.name].publish(msg)
        return ret
    
    def getPrimitive(self, s, a):
        # given state, action description and goal, return action representing left/right frequencies
        theta, phi, alpha = self.getAngles(s)
        diff_r = (np.linalg.norm(self.goal) - self.radius)/self.radius
        if a == "MOVE_TOWARDS":
            action = [math.pow(self.travel_gain * phi, 3), math.pow(self.travel_gain * theta, 3)]
        if a == "ANGLE_TOWARDS":
            action = [self.rotate_gain * np.cos(theta), -self.rotate_gain * np.cos(theta)]
        if a == "MOVE_AWAY":
            action = [-math.pow(self.travel_gain*theta, 3), -math.pow(self.travel_gain * phi, 3)]
        if a == "FORWARD":
            action = [4, 4]
        if a == "CIRCLE_C":
            action = [self.circle_gain*(1 + diff_r - np.sin(theta)), self.circle_gain*(1 - diff_r + np.sin(theta))]
        if a == "CIRCLE_CC":
            action = [self.circle_gain*(1 - diff_r + np.sin(phi)), self.circle_gain*(1 + diff_r - np.sin(phi))]
        if a == "ANGLE_SAME":
            action = [self.rotate_gain * np.cos(alpha), -self.rotate_gain * np.cos(alpha)]
        return action
    
    def getAngles(self, s):
        s = s.ravel()
        goal = self.goal.ravel()[:2] - s[:2] #relative position of goal
        ori = s[5]
        front_v = vector(ori)
        right_v = vector(ori - np.pi/2)

        relative_x = dot(unitVector(goal), right_v)
        relative_y = dot(unitVector(goal), front_v)

        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        theta = np.arctan(relative_y/relative_x) + buff 

        phi = -np.pi - theta if theta < 0 else np.pi - theta 

        relative_x = dot(vector(s[11]), right_v)
        relative_y = dot(vector(s[11]), front_v)
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        alpha = np.arctan(relative_y/relative_x) + buff 

        return theta, phi, alpha

    def checkConditions(self, s):
        # given self.prev['A'] and state (unraveled already), check that we've sufficiently executed primitive
        if self.prev['A'] == None:
            return True
        a = self.actionMap[self.prev['A']]
        s = np.array(s)
        theta, phi, _ = self.getAngles(s)
        self.counter -= 1
        #if a == "ANGLE_TOWARDS":
        #   return abs(theta - np.pi/2) < 2e-1
        return self.counter == 0

    def checkPhase(self, pos, blockPos, ori, blockOri, phase):
        if pos[-1] < .35:
            return (-3, 1)
        if phase == 1:
            if blockPos[2] < .3 and abs(blockPos[1]) < 1.2 and blockPos[0] > -.5:
                return (5, 0)
        if phase == 2:
            if pos[0] > .4:
                return (5, 1)
        return (0,0)
    
    def getAux(self, pos, prevPos, blockPos, prevBlock, ori, prevOri, phase, blockOri):
        if phase == 1:
            dist_r = (dist(prevPos, blockPos) - dist(pos, blockPos))

            block_r = blockPos[0] - prevBlock[0]
            prevVec = unitVector(vector(prevOri))
            vec = unitVector(vector(ori))
            goal = unitVector(blockPos[:2]-pos[:2])

            prevDot = dot(prevVec, goal)
            currDot = dot(vec, goal)
            ori_r = currDot - prevDot
            # ori_r = abs(prevOri - blockOri) - abs(ori - blockOri)
            # ori_r -= abs(blockOri)

            return ((2*block_r + dist_r + 2*ori_r - .1), 0)

        if phase == 2:
            goal = np.array([.40, blockPos[1], pos[2]])
            delta = dist(pos, goal)
            prevDelta = dist(prevPos, goal)
            dist_r = prevDelta - delta
            y_r = -abs(blockPos[1]-pos[1])

            return ((dist_r  + .15*y_r - .05 * abs(ori) - .1), 0)

    def unpack(self, prevS, s, double = False):
        if double:
            prevPos = np.array(prevS[:3])
            pos = np.array(s[:3])
            blockPos = np.array(s[4:7])
            prevBlock = np.array(prevS[4:7])
            ori = s[3]
            prevOri = prevS[3]
            blockOri = s[7]
        else:
            prevPos = np.array(prevS[:3])
            pos = np.array(s[:3])
            blockPos = np.array(s[6:9])
            prevBlock = np.array(prevS[6:9])
            ori = s[5]
            prevOri = prevS[5]
            blockOri = s[11]
        return prevPos, pos, blockPos, prevBlock, ori, prevOri, blockOri

    def rewardFunction(self, s, a):
        s = s.ravel()
        prevS = self.prev["S"].ravel()
        prevPos, pos, blockPos, prevBlock, ori, prevOri, blockOri = self.unpack(prevS, s)
        res = self.checkPhase(pos, blockPos, ori, blockOri, self.phase)
        if res[0] != 0:
            return res
        return self.getAux(pos, prevPos, blockPos, prevBlock, ori, prevOri, self.phase, blockOri)

    def receiveState(self, msg):    
        floats = vrep.simxUnpackFloats(msg.data)
        self.goal = np.array(floats[6:9]) # THIS IS A TEST
        fail = floats[-1]
        restart = 0
        floats = floats[:self.s_n]

        changeAction = self.checkConditions(floats)
        s = (np.array(floats)).reshape(1,-1)
        if changeAction:
            self.counter = self.period
            a = (self.sendAction(s))
            if type(self.prev["S"]) == np.ndarray:
                r, restart = self.rewardFunction(s, self.prev["A"])
                if r == 5:
                    print(" #### Phase ", self.phase, "  Complete!")
                    self.phase += 1
                self.agent.store(self.prev['S'], self.prev["A"], np.array([r]).reshape(1, -1), s, a, restart)
                self.currReward += r

            if self.trainMode:
                loss = self.agent.train()

            self.prev["S"] = s
            self.prev["A"] = a
        else:
            a = self.sendAction(s, changeAction)
            # SPECIAL CASE: since we execute one primitive for multiple time steps (with intermittent control updates), we need to store transitions/rewards when the agent fails out or succeeds
            if type(self.prev["S"]) == np.ndarray:
                r, restart = self.rewardFunction(s, self.prev["A"])
                if restart:
                    if r > 0: # we assume failure has rewards < 0
                        print(' #### Success!')
                    else:
                        print(' #### Dropped')
                    self.agent.store(self.prev['S'], self.prev["A"], np.array([r]).reshape(1, -1), s, a, restart)
                    self.currReward += r
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
        self.plotLoss(False, 'Loss Over Iterations w/ Moving Average', "Actor Loss over Iterations w/ Moving Average")
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
            x = range(len(self.agent.actorLoss))
            window = np.ones(int(15))/float(15)
            line = np.convolve(self.agent.actorLoss, window, 'same')
            plt.plot(x, line, 'r')
            plt.plot(x, self.agent.actorLoss)
            plt.title(title2)
            plt.show()
    
    def plotRewards(self):
        print(len(self.rewards))
        x = range(len(self.rewards))
        plt.plot(x, self.rewards)
        plt.title("Rewards Over Episodes w/ Moving Average")
        plt.legend()
        window= np.ones(int(15))/float(15)
        lineRewards = np.convolve(self.rewards, window, 'same')
        plt.plot(x, lineRewards, 'r')
        grid = True
        plt.show()
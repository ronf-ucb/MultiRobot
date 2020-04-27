#! /usr/bin/env python

from task import Task 
from task import distance
import numpy as np 
import torch 
import torch.nn as nn
import vrep
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt

class BoxTask(Task):
    def __init__(self, action):
        super(BoxTask, self).__init__(action)

        self.prev = {"S": None, "A": None}
        self.actionMap = {0: (-3,-2), 1:(-2,-3), 2:(-3,-3), 3:(2,3), 4:(3,3), 5:(3,2)}

        self.currReward = 0
        self.rewards = []
        self.phase = 1


    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.out_n = self.agent.out_n
        self.pubs = self.agent.pubs
        self.trainMode = self.agent.trainMode
        self.explore = self.agent.explore
        self.name = self.agent.name
        self.w_phase1 = self.vTrain['w_phase1']
        self.w_phase2 = self.vTrain['w_phase2']
        self.w_phase3 = self.vTrain['w_phase3']
 
    def sendAction(self, s):
        msg = Vector3()
        if self.a == "argmax":
            q = self.valueNet(s)
            i = np.random.random()
            if i < self.explore:
                index = np.random.randint(self.out_n)
                #print('random')
            else:
                q = q.detach().numpy()
                index = np.argmax(q)
                #print('determ')
           # print(self.actionMap[index])
            msg.x, msg.y = self.actionMap[index]
            action = np.array([index])
        if self.a == "p_policy":
            output = self.policyNet(torch.FloatTensor(s))
            action_mean = output[:, :int(out_n/2)]
            action_logstd = output[:, int(out_n/2):]
            action_std = torch.exp(action_logstd)
            action = (torch.normal(action_mean, action_std).detach().numpy()).ravel()
            msg.x, msg.y = (action[0], action[1])
        if self.a == "d_policy":
            output = self.policyNet(torch.FloatTensor(s))
            i = np.random.random()
            if i < self.explore[0]:
                #add in exploration 
                noise = torch.from_numpy(np.random.normal(0, self.explore[1], 2))
                output = output + noise
            output = output.float()
            msg.x = output[0][0]
            msg.y = output[0][1]
            self.pubs[self.name].publish(msg)
            return output
        self.pubs[self.name].publish(msg)
        return action.reshape(1,-1)

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
        if pos[-1] < .25:
            return -30

        if self.phase == 1:
            deltaX = blockPos[0] - pos[0]
            deltaY = blockPos[1] - pos[1]
            deltaOri = abs(ori - blockOri)
            if deltaX > 0 and deltaX < .5 and abs(deltaY) < .2 and deltaOri < np.pi / 4:
                print("### Phase ", self.phase, " completed")
                self.phase = 2
                return 20*(np.pi - abs(ori)) #reward getting there with good orientation
            
            goal = np.array([blockPos[0]-.2, blockPos[1], blockPos[2]])
            delta = distance(pos, goal)
            prevDelta = distance(prevPos, goal)

            prevVec = np.array([np.cos(prevOri), np.sin(prevOri)])
            vec = np.array([np.cos(ori), np.sin(ori)])
            goal = blockPos[:2]-pos[:2]
            goal = goal/np.linalg.norm(goal)

            prevDot = (prevVec/np.linalg.norm(prevVec)).dot(goal)
            dot = (vec/np.linalg.norm(vec)).dot(goal)
            #buff = 10 if prevDelta - delta > .5 else 0
            #straight =  2*(dot - prevDot)
            return (prevDelta - delta + 2*(dot - prevDot)) * self.w_phase1 #+ buff
        if self.phase == 2:
            if blockPos[2] < .4:
                print("### Phase ", self.phase, " completed")
                self.phase+=1
                return 80
            blockXVel = blockPos[0] - prevBlock[0]
            robXVel = pos[0] - prevPos[0]

            dist = distance(blockPos, pos)
            prevDist  = distance(prevBlock, prevPos)
            deltaOri = np.abs(prevS[11]) - np.abs(blockOri) + np.abs(prevS[5]) - abs(ori)

            buff = 10 if blockXVel > .1 and abs(blockOri) < np.pi/4 else 0 
            return (blockXVel + robXVel - .5*(dist - prevDist) + .05*deltaOri) *self.w_phase2 + buff 
        if self.phase == 3:
            if pos[0] > .6:
                print("### Success!")
                return 150
            goal = np.array([.80, blockPos[1], pos[2]])
            delta = distance(pos, goal)
            prevDelta = distance(prevPos, goal)
            buff = 10 if prevDelta - delta > .05 and abs(ori) < np.pi/4 else 0 
            return ((prevDelta - delta) - .15*abs(pos[1] - blockPos[1]) - .05*abs(ori)) * self.w_phase3 + buff
    


    def receiveState(self, msg):
        floats = vrep.simxUnpackFloats(msg.data)
        self.goal = np.array(floats[-4:-1])
        restart= floats[-1]
        s = (np.array(floats[:self.state_n])).reshape(1,-1)
        a = (self.sendAction(s))
        if type(self.prev["S"]) == np.ndarray:
            r = np.array(self.rewardFunction(s,a)).reshape(1,-1)
            #print(r)
            self.agent.store(self.prev['S'], self.prev["A"], r, s, a, restart)
            self.agent.dataSize += 1
            self.currReward += np.asscalar(r)
        self.prev["S"] = s
        self.prev["A"] = a
        s = s.ravel()
        if self.trainMode and self.agent.dataSize >= self.agent.batch_size:
            self.agent.train()
        if s[0] > .75:
            restart = 1 #success! 
        self.restartProtocol(restart)
        return 
    
    def restartProtocol(self, restart):
        if restart == 1:
            for k in self.prev.keys():
                self.prev[k] = None
            self.goal = 0
            if self.agent.trainIt > 0:
                self.agent.valueLoss.append((self.agent.avgLoss)/self.agent.trainIt)
                self.agent.avgLoss = 0  
                if self.a == 'p_policy' or self.a == 'd_policy':
                    self.agent.actorLoss.append((self.agent.avgActLoss)/self.agent.trainIt)
                    self.agent.avgActLoss = 0  
            self.agent.trainIt = 0
            self.rewards.append(self.currReward)
            self.currReward = 0
            self.phase = 1

    ######### POST TRAINING #########
    def postTraining(self):
        valueOnly = True if self.a == "argmax" else False
        self.plotLoss(valueOnly, "Value Loss over Iterations", "Actor Loss over Iterations")
        self.plotRewards()
        self.agent.saveModel()
    
    def plotLoss(self, valueOnly = False, title1 = "Critic Loss over Iterations", title2 = "Actor Loss over Iterations"):
        plt.plot(range(len(self.agent.valueLoss)), self.agent.valueLoss)
        plt.title(title1)
        plt.show()
        if not valueOnly:
            plt.plot(range(len(self.agent.actorLoss)), self.agent.actorLoss)
            plt.title(title2)
            plt.show()
    
    def plotRewards(self):
        plt.plot(range(len(self.rewards)), self.rewards)
        plt.title("Average rewards over Episodes")
        plt.legend()
        plt.show()
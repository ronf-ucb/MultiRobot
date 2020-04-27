#! /usr/bin/env python

#! /usr/bin/env python

from task import Task 
from task import distance as dist
import numpy as np 
import torch 
import torch.nn as nn
import vrep
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt
from boxTask import BoxTask

class BoxDoubleTask(BoxTask):
    def __init__(self, action):
        super(BoxDoubleTask, self).__init__(action)
        self.actionMap = {0: (-3,-2), 1:(-2,-3), 2:(-3,-3), 3:(2,3), 4:(3,3), 5:(3,2), 6:(0,0)}


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
        self.agents = self.agent.agents
 
    def sendAction(self, s):
        msgFirst = Vector3()
        msgSecond = Vector3()
        names = self.pubs.keys()
        names.sort()
        if self.a == "argmax":
            q = self.valueNet(s)
            i = np.random.random()
            if i < self.explore:
                index = np.random.randint(self.out_n)
            else:
                q = q.detach().numpy()
                index = np.argmax(q)
            assert len(names) == 2
            first = index // len(self.actionMap)
            second = index % len(self.actionMap)
            msgFirst.x, msgFirst.y = self.actionMap[first]
            msgSecond.x, msgSecond.y = self.actionMap[second]
            action = np.array([index])
            messages = [msgFirst, msgSecond]
        if self.a == "p_policy":
            pass
        if self.a == "d_policy":
            output = self.policyNet(torch.FloatTensor(s))
            i = np.random.random()
            if i < self.explore[0]:
                noise = torch.from_numpy(np.random.normal(0, self.explore[1], self.out_n))
                output = output + noise
            action = (output.float()[0]).detach().numpy()
            i = 0
            messages = []
            for n in names:
                u = self.agents[n]['u']
                msg = Vector3()
                if u == 2:
                    msg.x, msg.y = (action[i], action[i+1])
                if u == 3:
                    msg.x, msg.y, msg.z = (action[i], action[i+1], action[i+2])
                messages.append(msg)
                i += u
        for i in range(len(names)):
            self.pubs[names[i]].publish(messages[i])
        return action.reshape(1,-1)

    def rewardFunction(self, s, a):
        s = s.ravel()
        prevS = self.prev["S"].ravel()

        pPosOne = np.array(prevS[:3])
        posOne = np.array(s[:3])
        pPosTwo = np.array(prevS[12:15])
        posTwo = np.array(s[12:15])

        blockPos = np.array(s[6:9])
        prevBlock = np.array(prevS[6:9])
        blockOri = s[11]

        oriOne = s[5]
        pOriOne = prevS[5]
        oriTwo = s[17]
        pOriTwo = prevS[17]

        if posOne[-1] < .25 or posTwo[-1] < .25:
            return -30

        if self.phase == 1:
            d1 = np.subtract(blockPos, posOne)
            d2 = np.subtract(blockPos, posTwo)
            dOri1 = abs(oriOne - blockOri)
            dOri2 = abs(oriTwo - blockOri)
            if (d1[0]>0 and d1[0] < .6) and (d2[0] > 0 and d2[0] <.6) and (abs(d1[1]) < .5 and abs(d2[1]) < .5) and dOri1 < np.pi / 3 and dOri2 < np.pi/3:
                print("### Phase ", self.phase, " completed")
                self.phase = 2
                return 50
            goal = np.array([blockPos[0]-.2, blockPos[1], blockPos[2]])
            dDist = dist(pPosOne, goal) - dist(posOne, goal) + dist(pPosTwo, goal) - dist(posTwo, goal)
            dist_bots = dist(pPosOne, pPosTwo) - dist(posOne, posTwo)

            pV1 = np.array([np.cos(pOriOne), np.sin(pOriOne)])
            v1 = np.array([np.cos(oriOne), np.sin(oriOne)])
            pV2 = np.array([np.cos(pOriTwo), np.sin(pOriTwo)])
            v2 = np.array([np.cos(oriTwo), np.sin(oriTwo)])

            goalOne = blockPos[:2]-posOne[:2]
            goalOne = goalOne/np.linalg.norm(goalOne)
            goalTwo = blockPos[:2]-posTwo[:2]
            goalTwo = goalTwo/np.linalg.norm(goalTwo)

            dot = (v1/np.linalg.norm(v1)).dot(goalOne) - (pV1/np.linalg.norm(pV1)).dot(goalOne) + (v2/np.linalg.norm(v2)).dot(goalTwo) - (pV2/np.linalg.norm(pV2)).dot(goalTwo)
            return (dDist + 2*(dot) + 4*dist_bots) * self.w_phase1 

        if self.phase == 2:
            if blockPos[2] < .4:
                print("### Phase ", self.phase, " completed")
                self.phase+=1
                return 80
            blockXVel = blockPos[0] - prevBlock[0]
            vel = posOne[0] - pPosOne[0] + posTwo[0] - pPosTwo[0]

            d = dist(blockPos, posOne) + dist(blockPos, posTwo)
            prevDist  = dist(prevBlock, pPosOne) + dist(prevBlock, pPosTwo)
            deltaOri = np.abs(prevS[11]) - np.abs(blockOri) + np.abs(prevS[5]) - abs(oriOne) +np.abs(prevS[17]) - abs(oriTwo)

            buff = 10 if blockXVel > .1 and abs(blockOri) < np.pi/8 else 0 
            return (2*blockXVel + .8*vel - .5*(d - prevDist) + .05*deltaOri) *self.w_phase2 + buff 
        if self.phase == 3:
            if posOne[0] > .6 and posTwo[0] > .6:
                print("### Success!")
                return 150
            goal = np.array([.80, blockPos[1], posOne[2]])
            delta = dist(posOne, goal) + dist(posTwo, goal)
            prevDelta = dist(pPosOne, goal) + dist(pPosTwo, goal)

            return ((prevDelta - delta) - .15*(abs(posOne[1] - blockPos[1]) + abs(posTwo[1] - blockPos[1])) - .05*(abs(oriOne) -abs(oriTwo))- .1*blockOri) * self.w_phase3
    


    def receiveState(self, msg):
        floats = vrep.simxUnpackFloats(msg.data)
        self.goal = np.array(floats[-4:-1])
        restart= floats[-1]
        s = (np.array(floats[:self.state_n])).reshape(1,-1)
        a = (self.sendAction(s))
        if type(self.prev["S"]) == np.ndarray:
            r = np.array(self.rewardFunction(s,a)).reshape(1,-1)
            print(r)
            self.agent.store(self.prev['S'], self.prev["A"], r, s, a, restart)
            self.agent.dataSize += 1
            self.currReward += np.asscalar(r)
        self.prev["S"] = s
        self.prev["A"] = a
        s = s.ravel()
        if self.trainMode and self.agent.dataSize >= self.agent.batch_size:
            self.agent.train()
        if s[0] and s[12]> .75:
            restart = 1 #success! 
        self.restartProtocol(restart)
        return 
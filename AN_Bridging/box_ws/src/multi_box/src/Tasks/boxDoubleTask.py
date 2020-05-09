#! /usr/bin/env python

#! /usr/bin/env python

from task import Task 
from task import distance as dist
import numpy as np 
import torch 
import torch.nn as nn
import vrep
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt
from boxTask import BoxTask

class BoxDoubleTask(BoxTask):
    def __init__(self):
        super(BoxDoubleTask, self).__init__()
        self.actionMap = {0: (-3,-2), 1:(-2,-3), 2:(-3,-3), 3:(2,3), 4:(3,3), 5:(3,2), 6:(0,0)}
        self.s_n = 18


    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.pubs = self.agent.pubs
        self.trainMode = self.agent.trainMode
        self.w_phase1 = self.vTrain['w_phase1']
        self.w_phase2 = self.vTrain['w_phase2']
        self.w_phase3 = self.vTrain['w_phase3']
        self.agents = self.agent.agents
        self.name = self.agent.name
        rospy.Subscriber(self.agents[self.name]['sub'], String, self.receiveState, queue_size = 1) 
 
    def sendAction(self, s, w_s):
        #pass in the local state of agent and its name according to self.agents
        msg = Vector3()
        action, ret = self.agent.get_action(s, w_s)
        for (i, key) in enumerate(self.pubs.keys()):
            msg.x, msg.y = (action[i][0], action[i][1])
            self.pubs[key].publish(msg)
        return ret
    
    def rewardFunction(self, s_n, a):
        first, second = self.splitState(s_n.ravel().tolist())
        fPrev, sPrev  = self.splitState(self.prev['S'].ravel().tolist())
        
        prevPos, pos, blockPos, prevBlock, ori, prevOri, blockOri = self.unpack(fPrev, first)
        res1 = self.checkPhase(pos, blockPos, ori, blockOri, self.phase)
        r_1 = self.getAux(pos, prevPos, blockPos, prevBlock, ori, prevOri, self.phase)

        prevPos, pos, blockPos, prevBlock, ori, prevOri, blockOri = self.unpack(sPrev, second)
        res2 = self.checkPhase(pos, blockPos, ori, blockOri, self.phase)   
        r_2 = self.getAux(pos, prevPos, blockPos, prevBlock, ori, prevOri, self.phase)
        
        fail1 = (res1[0] == -3)
        fail2 = (res2[0] == -3)
        success = self.phase == 3 and (res1[0] == 5 and res2[0] == 5)
        restart = 1 if (fail1 or fail2 or success) else 0

        changePhase = (res1[0] == 5 and res2[0] == 5)
        r_1 = res1[0] if fail1 or changePhase else r_1[0]
        r_2 = res2[0] if fail2 or changePhase else r_2[0] 
        
        if changePhase:
            print(" ## Phase: ", self.phase, " complete! ##")
            self.phase += 1

        return ([r_1, r_2], restart)

    
    def splitState(self, s):
        first = s[:12]
        second = s[12:18] + s[6:12]
        return first, second


    def receiveState(self, msg):
        floats = vrep.simxUnpackFloats(msg.data)
        self.goal = np.array(floats[-4:-1])
        fail = floats[-1]
        restart = 0
        floats = floats[:self.s_n]
        first, second = self.splitState(floats)

        if self.phase == 1:
            first.append(0)
            second.append(0)
            floats.append(0)
        else:
            first.append(1)
            second.append(1)
            floats.append(1)
        
        s = (np.array(floats)).reshape(1,-1)
        first = torch.FloatTensor(first).view(1,-1)
        second = torch.FloatTensor(second).view(1,-1)
        a = (self.sendAction(s, [first, second]))
        if type(self.prev["S"]) == np.ndarray:
            r, restart = self.rewardFunction(s,a)   
            self.agent.store(self.prev['S'], self.prev["A"], r, s, a, restart)
            self.currReward += sum(r)
        self.prev["S"] = s
        self.prev["A"] = a
        s = s.ravel()
        if self.trainMode:
            self.agent.train()
        self.restartProtocol(restart or fail)
        return 
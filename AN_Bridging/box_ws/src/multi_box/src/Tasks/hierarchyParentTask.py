#! /usr/bin/env python

from task import distance as dist 
from task import Task, dot, unitVector
import numpy as np 
import torch 
import torch.nn as nn
import vrep
import rospy
from std_msgs.msg import String, Int8
from moveTask import MoveTask
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt

'''Class for any parent in the hierarchy. Inherits from goalSetTask and only changes reward and receiveState functions'''

class ParentTask(MoveTask):
    def __init__(self, action):
        super(ParentTask, self).__init__(action)

        self.prev = {"S": None, "A": None}
        rospy.Subscriber('/restart', Int8, self.restartCall, queue_size = 1)
        rospy.Subscriber('/parent', String, self.receiveGoal, queue_size = 1) #SOMEHOW FIX THIS

        self.currReward = 0
        self.rewards = []
        self.goal = 0
        self.sigma = .3
        self.distances = []

    def rewardFunction(self, s_n, a):
        pass #this depends on which level of hierarchy we are at. 

    def receiveState(self, msg):
        if self.currIt == self.c:
            s = np.array(vrep.simxUnpackFloats(msg.data))
            s_cat = np.hstack((s, self.goal))
            finish = 0  

            self.prevIt = self.currIt 
            a = (self.sendAction(s_cat))

            if type(self.prev["S"]) == np.ndarray:
                r = np.array(self.rewardFunction(s, self.prev['A'])).reshape(1,-1)
                self.agent.store(self.prev['S'].reshape(1,-1), self.prev["A"], r, s_cat.reshape(1,-1), None, finish)
                self.agent.dataSize += 1
                self.currReward += np.asscalar(r)

            self.prev["S"] = s_cat
            self.prev["A"] = a.reshape(1,-1)
            s = s.ravel()
            if self.trainMode and self.agent.dataSize >= self.agent.batch_size:
                self.agent.train()
            self.currIt = 0
        self.currIt += 1
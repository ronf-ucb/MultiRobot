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
from hierarchyParentTask import HierarchyParentTask

'''Class for any parent in the hierarchy. Inherits from goalSetTask and only changes reward and receiveState functions'''

class PositionTask(HierarchyParentTask):
    def __init__(self, action):
        super(PositionTask, self).__init__(action)

        self.prev = {"S": None, "A": None}
        rospy.Subscriber('/restart', Int8, self.restartCall, queue_size = 1)
        rospy.Subscriber('/parent', String, self.receiveGoal, queue_size = 1) #SOMEHOW FIX THIS

        self.c = 15 #change this to a tuneable parameter from an "agent" file
        self.currIt = self.c

    def rewardFunction(self, s_n, a):
        #say the goal has three parts: box x, box y, agent x
        #our state consists of everything regarding this agent: (x,y,z) and (pitch, roll, yaw) plus (box x, box y)
        #incorporate two parts: how well does the child follow our goal   
        #                       how much closer we are to achieving our parent's goal
        child_r = 1/(dist(a, s_n[:2])) #how close our child got to our goal
        parent_r = abs(self.goal[0] - self.prev['S'][0]) - abs(self.goal[0] - s_n[0])
        parent_r += dist(self.goal[:2], self.prev['S'][6:8]) - dist(self.goal[:2], s_n[6:8])
        return child_r + parent_r



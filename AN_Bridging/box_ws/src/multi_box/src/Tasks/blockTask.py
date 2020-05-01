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
        #we have no "goal"...we are the top of the tree! 
        #our state consists of: x of agent 1, x of agent 2, (x, y, z) and (roll, pitch, yaw) of box
        #incorporate two parts: how well do our children follow our goal
        #                       are we closer to getting to where we want?



#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header, Int16
import numpy as np

from scipy import signal, stats
import matplotlib.pyplot as plt
import math
from geometry_msgs.msg import Polygon, Point32, Vector3, Pose
from nav_msgs.msg import Odometry


# import vrep
import matplotlib.pyplot as plt
import sys

import time

plan_A = [2, 2]
plan_B = [3, 1]
plan_C = [3, 2]

class Overall_controll:
    def __init__(self, plan):

        self.sub_state = rospy.Subscriber("/cockroach_state_actual", Pose, self.getState, queue_size=1)
        self.pub_state = rospy.Publisher("/cockroach_state", Pose, queue_size = 1)
        
        self.plan = plan

        self.state_list_track = [0, 0, 0]
        self.state_list_follow = [0, 0, 0]

    def getState(self, msg):
        self.state_list_track[0] = msg.position.x
        self.state_list_track[1] = msg.position.y
        self.state_list_track[2] = msg.position.z
        self.state_list_follow[0] = msg.orientation.x
        self.state_list_follow[1] = msg.orientation.y
        self.state_list_follow[2] = msg.orientation.z
        print("track : ", self.state_list_track)
        print("follow : ", self.state_list_follow)

        self.controllerCB()

    def controllerCB(self):
        ########### tracker controller
        num_track = 0
        for i in self.state_list_track:
            if i == 1:
                num_track += 1
        diff_track = self.plan[0] - num_track

        count_track = 0
        if diff_track > 0:
            for i in self.state_list_track:
                if diff_track == 0:
                    break
                elif i == 0:
                    diff_track -= 1
                    self.state_list_track[count_track] = 1
                count_track += 1
        else:
            for i in self.state_list_track:
                if diff_track == 0:
                    break
                elif i == 0:
                    diff_track += 1
                    self.state_list_track[count_track] = 0
                count_track += 1



        ########### follower controller
        num_follow = 0
        for i in self.state_list_follow:
            if i == 1:
                num_follow += 1
        diff_follow = self.plan[1] - num_follow

        count_follow = 0
        if diff_follow > 0:
            for i in self.state_list_follow:
                if diff_follow == 0:
                    break
                elif i == 0:
                    diff_follow -= 1
                    self.state_list_follow[count_follow] = 1
                count_follow += 1
        else:
            for i in self.state_list_follow:
                if diff_follow == 0:
                    break
                elif i == 0:
                    diff_follow += 1
                    self.state_list_follow[count_follow] = 0
                count_follow += 1

        ############ publish required state to robots
        pos = Pose()
        pos.position.x = self.state_list_track[0]
        pos.position.y = self.state_list_track[1]
        pos.position.z = self.state_list_track[2]

        pos.orientation.x = self.state_list_follow[0]
        pos.orientation.y = self.state_list_follow[1]
        pos.orientation.z = self.state_list_follow[2]

        self.pub_state.publish(pos)



if __name__=="__main__":
    rospy.init_node("overall_controller")
    # Overall_controll(plan_A)
    # Overall_controll(plan_B)
    Overall_controll(plan_C)
    rospy.spin()

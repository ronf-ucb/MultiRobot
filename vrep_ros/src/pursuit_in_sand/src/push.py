#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header, Int16
import numpy as np

import matplotlib.pyplot as plt
import math
from geometry_msgs.msg import Polygon, Point32, Vector3, Pose
from nav_msgs.msg import Odometry


# import vrep
import matplotlib.pyplot as plt
import sys

import time

LSignalName = "CycleLeft"
RSignalName = "CycleRight"
# BaseFreq = -1
BaseFreq = -2
coef = 1.1
# coef = 1.6

state = 0
obj = None


class Pursuit:
    def __init__(self, leftName, rightname):

        self.pos0 = Point32()
        self.pos1 = Point32()
        self.pos2 = Point32()
        # prepare the sub and pub
        self.sub_path = rospy.Subscriber("/cockroachPath_push", Vector3, self.getPath, queue_size=1)
        self.sub_push_pos = rospy.Subscriber("/cockroachPos_push", Point32, self.getPushPos, queue_size=1)
        self.sub_pull_pos = rospy.Subscriber("/cockroachPos_pull", Point32, self.getPullPos, queue_size=1)
        self.pub_vel = rospy.Publisher("/cockroachVel_push", Vector3, queue_size = 1)
        self.pub_pathNum = rospy.Publisher("/pathNum_push", Vector3, queue_size = 1)


        self.LSignalName = leftName
        self.RSignalName = rightname
        self.stop_sig = 0

        self.LCycleFreq = BaseFreq
        self.RCycleFreq = BaseFreq

        self.goal_received = False
        self.goal_reached = False

        self.rPose = None

        self.last_theta = 0
        self.path_num = 0

        self.pos = Point32()     # robot position
        self.ori = None     # robot orientation

        self.pull_pos = Point32()     # another robot position
        self.pull_ori = None     # another robot orientation

        self.path = Vector3()      # destination points

        # self.kp = 0.3
        self.kp = 2
        # self.kd = -0.3
        self.kd = 2
        self.ki = 0.01

        ## Now Revising the PID Coefficient
        self.kp = 1
        self.kd = 0
        self.ki = 0


        self.eta = None

        self.radiu = 0.02      # the dist from destination point that robot stop
        # self.radiu = 0.0      # the dist from destination point that robot stop
        self.flag = False   # check if robot reach the destination point

        # for test :: to get the handle
        self.cube = None
        self.cube1 = None

        self.last_pose = Point32()
        self.help_request = False

        # coef for angle in the middle. 
        # self.coef_theta_diff = 0.25
        # self.coef_theta_diff = 0
        self.coef_theta_diff = 2
    
    def get_stop_sig(self,msg):
        self.stop_sig = msg.x
        print('get msg')


    def getPullPos(self, msg):
        """
        implement localization and getPath
        """

        self.pull_pos = msg
        self.pull_ori = msg.z

    def getPushPos(self, msg):
        """
        implement localization and getPath
        """

        self.pos = msg
        self.ori = msg.z


            
        self.controller()

    def getPath(self, msg):
        """
        get the path which need to track    TODO
        """
        # destination (for first step)
        self.last_pose.x = self.path.x
        self.last_pose.y = self.path.y

        self.path.x = msg.x
        self.path.y = msg.y



    def getEta(self, pose):     # TODO : refer to test_controller
        """
        get the eta between robot orientation and robot position to destination     TODO: how to get the delta orientation
        :param pose: tracking position
        :return: eta
        """
        vector_x = np.cos(self.ori) * (pose.x - self.pos.x) + np.sin(self.ori) * (pose.y - self.pos.y)
        vector_y = -np.sin(self.ori) * (pose.x - self.pos.x) + np.cos(self.ori) * (pose.y - self.pos.y)
        eta = math.atan2(vector_y, vector_x)
        # vector_x = (pose.x - self.pos.x)
        # vector_y = (pose.y - self.pos.y)
        # eta =math.atan(math.tan(math.atan2(vector_y, vector_x) - self.ori))
        return eta

    def if_goal_reached(self, pose):
        """
        check iff dist between robot and destination is less than limit
        :return: True / False
        """
        dx = self.pos.x - pose.x
        dy = self.pos.y - pose.y
        self.dist = math.sqrt(dx ** 2 + dy ** 2)
        self.total_theta = 0
        return self.dist < self.radiu

    def controller(self):
        # theta = self.getEta(self.path[self.path_num])
        theta = self.getEta(self.path)
        if theta < 0:
            if theta < -1.6:
                theta = -3.14 - theta
        else:
            if theta > 1.6:
                theta = 3.14 - theta
        
        # theta_diff is the angle in the middle
        theta_diff = self.getEta(self.pull_pos)
        if theta_diff < 0:
            if theta_diff < -1.6:
                theta_diff = -3.14 - theta_diff
        else:
            if theta_diff > 1.6:
                theta_diff = 3.14 - theta_diff


        if not self.if_goal_reached(self.path):
            self.LCycleFreq = (BaseFreq + (self.kp * theta + (theta - self.last_theta) * self.kd + self.ki * self.total_theta))*coef + self.coef_theta_diff * theta_diff
            self.RCycleFreq = (BaseFreq - (self.kp * theta + (theta - self.last_theta) * self.kd + self.ki * self.total_theta))*coef - self.coef_theta_diff * theta_diff
        
            # print("goal_num : ", self.path_num)
            # print("Error : ", theta)
            # print("goal : ", self.path)
            # print("self.L_vel : ", self.LCycleFreq)
            # print("self.R_vel : ", self.RCycleFreq)
            if  self.stop_sig == 1:
                self.LCycleFreq = 0
                self.RCycleFreq = 0
        else:
            print("goal reached !!")
            self.LCycleFreq = 0
            self.RCycleFreq = 0
            if self.path_num == 15 or self.stop_sig == 1:
                print("Bingo !!!")
            else:
                self.path_num += 1
        if self.dist < 0.08:
            self.LCycleFreq = 1*self.LCycleFreq
            self.RCycleFreq = 1*self.RCycleFreq
        self.last_theta = theta
        self.total_theta += theta


        vel = Vector3()
        vel.x = self.LCycleFreq
        vel.y = self.RCycleFreq

        pathMsg = Vector3()
        pathMsg.x = self.path_num

        self.pub_pathNum.publish(pathMsg)
        self.pub_vel.publish(vel)


# for dynamically allocate
# class state_prepare:
#     def __init__(self):
#         self.sub_state = rospy.Subscriber("/cockroach_state", Pose, self.getState, queue_size=1)

#     def getState(self, msg):
#         global state
#         global obj
#         state = msg.orientation.x
#         print(state)
#         if state:
#             obj = Pursuit(LSignalName, RSignalName)
#         else:
#             print("no move")


if __name__=="__main__":
    rospy.init_node("cockroachRun_2")
    # state_prepare()
    Pursuit(LSignalName, RSignalName)
    rospy.spin()

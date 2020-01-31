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

# BaseFreq = 1
# BaseFreq = -1
BaseFreq = -2
# coef = 1
coef = 1.2

state = 0
obj = None

class Pursuit:
    def __init__(self, leftName, rightname):


        self.pos0 = Point32()
        self.pos1 = Point32()
        self.pos2 = Point32()
        # prepare the sub and pub
        self.sub_path = rospy.Subscriber("/cockroachPath_pull", Vector3, self.getPath, queue_size=1)
        self.sub_pull_pos = rospy.Subscriber("/cockroachPos_pull", Point32, self.getPullPos, queue_size=1)
        self.sub_push_pos = rospy.Subscriber("/cockroachPos_push", Point32, self.getPushPos, queue_size=1)
        self.pub_vel = rospy.Publisher("/cockroachVel_pull", Vector3, queue_size = 1)
        self.pub_pathNum = rospy.Publisher("/pathNum_pull", Vector3, queue_size = 1)
        self.pub_stop = rospy.Publisher("/cockroachFlag_pull",Vector3,queue_size = 1)

        self.pub_state = rospy.Publisher("/cockroach_state_actual", Pose, queue_size = 1)

        self.stop_sig = Vector3()
        self.stop_sig.x = 0
        self.pub_stop.publish(self.stop_sig)


        self.LSignalName = leftName
        self.RSignalName = rightname

        self.LCycleFreq = BaseFreq
        self.RCycleFreq = BaseFreq

        self.goal_received = False
        self.goal_reached = False

        self.rPose = None

        self.last_theta = 0
        self.path_num = 0

        self.pos = Point32()     # robot position
        self.ori = None     # robot orientation

        self.push_pos = Point32()     # another robot position
        self.push_ori = None     # another robot orientation

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

        self.radiu = 0.05      # the dist from destination point that robot stop
        # self.radiu = 0.01      # the dist from destination point that robot stop
        self.flag = False   # check if robot reach the destination point

        # for test :: to get the handle
        self.cube = None
        self.cube1 = None

        # coef for angle in the middle. 
        # self.coef_theta_diff = -0.25
        # self.coef_theta_diff = 0
        self.coef_theta_diff = -2


    def getPushPos(self, msg):
        """
        implement localization and getPath
        """

        self.push_pos = msg
        self.push_ori = msg.z

    def getPullPos(self, msg):
        """
        implement localization and getPath
        """

        self.pos = msg
        self.ori = msg.z
        
        self.controller()

        # state_curr = Pose()
        # state_curr.position.x = state
        # self.pub_state.publish(state_curr)
        # time.sleep(1)
        # self.controller()

    def getPath(self, msg):
        """
        get the path which need to track    TODO
        """
        # destination (for first step)
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

    def getMiddleEta(self, pose):     # TODO : refer to test_controller
        """
        get the eta between robot orientation and robot position to destination     TODO: how to get the delta orientation
        :param pose: tracking position
        :return: eta
        """
        vector_x = np.cos(self.push_ori) * (pose.x - self.push_pos.x) + np.sin(self.push_ori) * (pose.y - self.push_pos.y)
        vector_y = -np.sin(self.push_ori) * (pose.x - self.push_pos.x) + np.cos(self.push_ori) * (pose.y - self.push_pos.y)
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
        dist = math.sqrt(dx ** 2 + dy ** 2)
        self.total_theta = 0
        return dist < self.radiu

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
        theta_diff = self.getMiddleEta(self.pos)
        if theta_diff < 0:
            if theta_diff < -1.6:
                theta_diff = -3.14 - theta_diff
        else:
            if theta_diff > 1.6:
                theta_diff = 3.14 - theta_diff
        

        if not self.if_goal_reached(self.path):
            self.LCycleFreq = (BaseFreq + (self.kp * theta + (theta - self.last_theta) * self.kd + self.ki * self.total_theta))*coef + self.coef_theta_diff * theta_diff
            self.RCycleFreq = (BaseFreq - (self.kp * theta + (theta - self.last_theta) * self.kd + self.ki * self.total_theta))*coef - self.coef_theta_diff * theta_diff
        
            # print("Yoooooooooooooooo")
            # print("goal_num : ", self.path_num)
            # print("Error : ", theta)
            # print("goal : ", self.path)
            print("self.L_vel : ", self.LCycleFreq)
            print("self.R_vel : ", self.RCycleFreq)
        else:
            print("goal reached !!")
            if self.path_num == 30:
                self.LCycleFreq = 0
                self.RCycleFreq = 0
                print("Bingo !!!")
                self.stop_sig.x = 1
                self.pub_stop.publish(self.stop_sig)
            else:
                self.path_num += 1

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
#         self.pub_state = rospy.Publisher("/cockroach_state_actual", Pose, queue_size = 1)


#     def getState(self, msg):
#         global state
#         global obj
#         state = msg.position.x
#         print(state)
#         if state:
#             obj = Pursuit(LSignalName, RSignalName)
#         else:
#             print("no move")



if __name__=="__main__":
    rospy.init_node("cockroachRun_1")
    
    # pub_state_init = rospy.Publisher("/cockroach_state_actual", Pose, queue_size = 1)
    # state_init = Pose()
    # state_init.position.x = state
    # state_init.position.y = 0
    # state_init.position.z = 0
    # state_init.orientation.x = 0
    # state_init.orientation.y = 0
    # state_init.orientation.z = 0
    # pub_state_init.publish(state_init)
    
    # sp = state_prepare()
    Pursuit(LSignalName, RSignalName)
    rospy.spin()

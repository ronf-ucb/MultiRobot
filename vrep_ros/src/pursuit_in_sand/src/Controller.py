#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Vector3
import time

sub_front = ["/cockroachPos_pull","/cockroachOri_pull","/cockroachVelSub_pull"]
sub_back = ["/cockroachPos_push","/cockroachOri_push","/cockroachVelSub_push"]
pub_front = "/cockroachVelPub_pull"
pub_back = "/cockroachVelPub_push"

class Robot():
    def __init__(self, robotname):
    
        self.action_space = [i for i in range(5)]
        self.observation_space = [0 for i in range(9)]

        ###########选择是哪个机器人################
        if robotname == "Front":
           rospy.Subscriber(sub_front[0],Vector3,self.get_position,queue_size = 1)
           rospy.Subscriber(sub_front[1],Vector3,self.get_oritation,queue_size = 1)
           rospy.Subscriber(sub_front[2],Vector3,self.get_velocity,queue_size = 1)
           self.VelPub = rospy.Publisher(pub_front,Vector3,queue_size = 1)
        else:
           rospy.Subscriber(sub_back[0],Vector3,self.get_position,queue_size = 1)
           rospy.Subscriber(sub_back[1],Vector3,self.get_oritation,queue_size = 1)
           rospy.Subscriber(sub_back[2],Vector3,self.get_velocity,queue_size = 1)
           self.VelPub = rospy.Publisher(pub_back,Vector3,queue_size = 1)

        self.action_space = ['goStraight','turnRight','turnLeft','goBack','stopRobot']
        self.observation_space = [0 for i in range(9)]
        self.CycleFreqL = 1
        self.CycleFreqR = 1
        
        
        

    #ALL DRIVING FUNCTIONS ARE CALLED DIRECTLY FROM THE AGENT. NO NEED TO IMPLEMENT ROS FUNCTINALITY
    def goStraight(self):
        self.CycleFreqL = 4
        self.CycleFreqR = 4
        self.sendSignal()

    def turnRight(self):
        self.CycleFreqL = 4
        self.CycleFreqR = 2
        self.sendSignal()

    def turnLeft(self):
        self.CycleFreqL = 2
        self.CycleFreqR = 4
        self.sendSignal()

    def goBack(self):
        self.CycleFreqL = -4
        self.CycleFreqR = -4
        self.sendSignal()

    def stopRobot(self):
        self.CycleFreqL = 0
        self.CycleFreqR = 0
        self.sendSignal()

    def sendSignal(self):
    ########传输机器人的速度############
        v = Vector3()
        v.x = self.CycleFreqL
        v.y = self.CycleFreqR
        self.VelPub.publish(v)
        
        
    def get_position(self,msg):
  #########得到机器人的坐标,1*3向量##################
        self.observation_space[0] = msg.x
        self.observation_space[1] = msg.y
        self.observation_space[2] = msg.z
    
    def get_oritation(self):
    #######得到机器人偏移角，1*3向量##################
        self.observation_space[3] = msg.x
        self.observation_space[4] = msg.y
        self.observation_space[5] = msg.z
    
    
    def get_velocity(self):
    #######得到机器人速度，1*3向量####################
        self.observation_space[6] = msg.x
        self.observation_space[7] = msg.y
        self.observation_space[8] = msg.z
    
    def step(self,action):
    #######传入一个数字类型，调用front，left等函数，最后返回下一个状态的state space，done，############
    #######重要，与环境交互的重要函数###################################
        act = [self.goStraight,self.turnLeft,self.turnRight,self.goBack,self.stopRobot]
        act[action]()
        time.sleep(5)
    #################done如何判断###############
    
    #################done如何判断###############
        return self.observation_space,False
        

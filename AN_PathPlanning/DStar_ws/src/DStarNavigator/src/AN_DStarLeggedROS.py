#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from AN_DStarAgentROS import DStarAgent

class LeggedDStar(DStarAgent):
    def __init__(self, leftName, rightName):
        super(LeggedDStar, self).__init__() #call parent class init
        #ROS: publish to topic connected to V-Rep
        self.pub = rospy.Publisher("/frequency", Vector3, queue_size = 1)
        #robot movement
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
        v = Vector3()
        v.x = self.CycleFreqL
        v.y = self.CycleFreqR
        self.pub.publish(v)

rospy.init_node("Agent", anonymous=True)
agent = LeggedDStar(LSignalName, RSignalName)
agent.prepare()
agent.policy()

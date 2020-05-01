#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt

from Networks.network import Network

class Agent(object):
    def __init__(self, params, name, task):
        self.name = name
        self.task = task
        self.vPars = params['valPars']
        self.vTrain = params['valTrain']
        self.agents = params['agents']
        rospy.Subscriber("/finished", Int8, self.receiveDone, queue_size = 1)
        self.pubs = {}
        for key in self.agents.keys():
            bot = self.agents[key]
            self.pubs[key] = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)

        self.batch_size     = self.vTrain['batch']
        self.discount       = self.vTrain['gamma']
        self.prob           = self.vPars['prob']
        self.trainMode      = self.vPars['trainMode']
        self.explore        = self.vTrain['explore']
        self.dataSize       = 0 #number of data tuples we have accumulated so far
        self.stop           = False
        self.trainIt        = 0
        self.totalSteps     = 0
        self.load           = self.vPars['load']
    
    def receiveDone(self, message):
        if message.data  == 1: #all iterations are done. Check manager.py
            self.stop = True
        if message.data == 2: #timed out. Check manager.py
            self.task.restartProtocol(restart = 1)

    def store(self, s, a, r, sprime, aprime = None, restart = 0):
        self.exp.push((s, a, r, sprime, aprime, restart))
    
    def train(self):
        pass 
    
    def saveModel(self):
        pass


    

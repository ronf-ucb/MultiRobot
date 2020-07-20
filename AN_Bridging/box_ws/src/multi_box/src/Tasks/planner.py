#! /usr/bin/env python

from task import Task, unitVector, dot, vector
from task import distance as dist
import numpy as np 
import math
import rospy
import torch 
import torch.nn as nn
import vrep
import time
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt
from hierarchyTask import HierarchyTask 
from box_slope_task import BoxSlopeTask
from Algs.doubleQ import DoubleQ
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

load_paths = {
    'slope_push': ['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/slope_push.txt'],
    'push_in_hole': ['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/push_in_hole.txt',
                    '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/push_in_hole2.txt'],
    'cross': ['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/cross.txt'],
    'reorient': ['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/reorient.txt',
                '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/reorient2.txt'],
    'push_towards':['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/push_towards_reduced_state.txt',
                    '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/push_towards_reduced_state2.txt',
                    '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/push_towards_reduced_state3.txt']
}

data_paths = {
    'slope_push': '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/SLOPE_PUSH_state_data.txt',
    'push_in_hole': '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/PUSH_IN_HOLE_state_data.txt',
    'cross': '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/CROSS_state_data.txt',
    'push_towards': '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/PUSH_TOWARDS_state_data.txt',
    'reorient': '/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_Bridging/REORIENT_state_data.txt',
}

class Planner(object):
    def __init__(self, network_parameters, name):
        self.fail = rospy.Publisher("/restart", Int8, queue_size = 1)
        self.primitive_state_n = {'slope_push': 6, 'push_in_hole': 7, 'reorient': 7, 'cross': 7, 'push_towards': 6}
        self.primitive_action_n = {'slope_push': 4, 'push_in_hole': 5, 'reorient': 5, 'cross': 4, 'push_towards': 5}
        self.primitive_nets = {}
        self.primitive_tasks = {}

        self.add_task(HierarchyTask, self.primitive_tasks, 'push_in_hole', reset=True)
        self.add_task(HierarchyTask, self.primitive_tasks, 'cross', reset=True)
        self.add_task(HierarchyTask, self.primitive_tasks, 'reorient', reset=True)
        self.add_task(HierarchyTask, self.primitive_tasks, 'push_towards', reset=True)
        self.add_task(BoxSlopeTask, self.primitive_tasks, 'slope_push', reset=False)

        self.agents = network_parameters['agents']
        self.pubs = OrderedDict()
        for key in self.agents.keys():
            bot = self.agents[key]
            self.pubs[key] = rospy.Publisher(bot['pub'], Vector3, queue_size = 1)
        self.name = name
            
        for key in self.primitive_state_n.keys():
            neurons = network_parameters['valPars']['neurons']
            new_neurons = [neurons[i] for i in range(len(neurons))]
            new_neurons[0] = self.primitive_state_n[key]
            new_neurons[-1] = self.primitive_action_n[key]
            network_parameters['valPars']['neurons'] = tuple(new_neurons)
            network_parameters['valPars']['mu'] = torch.Tensor([0 for i in range(self.primitive_state_n[key])])
            network_parameters['valPars']['std'] = torch.Tensor([1 for i in range(self.primitive_state_n[key])])
            ensemble = [DoubleQ(network_parameters, name, self, model) for model in load_paths[key]]
            self.primitive_nets[key] = ensemble
        rospy.Subscriber(self.agents[self.name]['sub'], String, self.receiveState, queue_size = 1) 
        self.changePoint = rospy.Publisher('/phaseChange', Int8, queue_size=1)

        self.default_prim = 'reorient'
        self.use_classifier = True
        self.data = []
        if self.use_classifier:
            self.train_classifier()
            self.train_model()

        while(True):
            x = 1+1
    
        
    def add_task(self, task_type, primitive_tasks, primitive_name, reset):
        curr = task_type()
        if reset:
            curr.resetPrimitive(primitive_name.upper())
        primitive_tasks[primitive_name] = curr 
        return 
    
    def initAgent(self, agent):
        pass

 
    def train_classifier(self):
        # From a dict of primitives -> paths, get the relevant data 
        # Train QDA using sklearn
        self.id_to_primitive = {}
        states = None 
        ids = None 
        for i, key in enumerate(data_paths.keys()):
            path = data_paths[key]
            self.id_to_primitive[i] = key
            s = np.loadtxt(path)
            states = s if i == 0 else np.vstack((states, s))
            idx = np.zeros(s.shape[0]) + i  
            ids = idx if i == 0 else np.hstack((ids, idx))
        self.classifier = KNN(n_neighbors = 200)
        self.classifier.fit(states, ids)
        return 
    
    def train_model(self):
        # For each of the data paths (both success and failure)
        # Find a 

    def get_reward_from_feature(self, feature, primitive):
        rewards = {'slope_push': dist(feature[5:8], np.zeros(3)), 
                   'push_in_hole': dist(feature[5:7], np.zeros(2)), 
                   'reorient': 10*(abs(feature[4]) + abs(feature[8])), 
                   'cross': dist(feature[5:7], np.zeros(2)), 
                   'push_towards': dist(feature[5:7], np.zeros(2))}
        return rewards[primitive]

    def choose_next_primitive(self, feature, next_feature):
        distances, indices = self.classifier.kneighbors(feature.reshape(1, -1)) 
        distances_next, indices_next = self.classifier.kneighbors(next_feature.reshape(1, -1))

        curr_sum = np.sum(distances)
        next_sum = np.sum(distances_next)
        normal = max(curr_sum, next_sum)
        curr_gain = self.get_reward_from_feature(feature, self.actionMap[self.classifier.predict(feature.reshape(1,-1))])
        next_gain = self.get_reward_from_feature(feature, self.actionMap[self.classifier.predict(next_feature.reshape(1,-1))])
        curr_expected = (curr_sum / normal) * curr_gain 
        next_expected = (next_sum / normal) * next_gain
        change = next_expected > curr_expected
        if change:
            msg = Int8()
            msg.data = 1
            self.changePoint.publish(msg)
        return self.id_to_primitive[np.asscalar(self.classifier.predict(feature.reshape(1,-1)))]


    def sendActionForPlan(self, states, phase):
        if self.use_classifier:
            next_prim = self.choose_next_primitive(states['feature'], states['feature_next'])
            if next_prim == 'slope_push':
                s = states['slope']
            else:
                s = states['flat']
        else:
            if phase == 1:
                s = states['slope']
                next_prim = 'slope_push'
            else:
                s = states['flat']
                if phase == 6:
                    next_prim = 'cross' 
                elif phase == 5 :
                    next_prim = 'push_in_hole'
                else:
                    if self.isValid(s, 'reorient'):
                        next_prim = 'reorient'
                    else:
                        next_prim = 'push_towards'

        task = self.primitive_tasks[next_prim]
        task.box_height = .5 
        task.goal = s[:2]
        self.sendAction(s, task, next_prim)
        return 
    
    def sendAction(self, s, task, next_prim):
        # Pick between ensemble of policies 
        model_index = np.random.randint(len(self.primitive_nets[next_prim]))
        net = self.primitive_nets[next_prim][model_index]
        task.agent = net
        task.pubs = self.pubs
        task.name = self.name

        changeAction = task.checkConditions(s, task.prev['A'])

        if changeAction and next_prim != 'slope_push':
            print('Curr primitive: ', next_prim)
            task.counter = task.period

        if next_prim == 'slope_push':
            print('Curr primitive: ', next_prim)
            index = task.sendAction(s)
        else:
            index = task.sendAction(s, changeAction) 
        task.prev['A'] = index
        return index
    
    def isValid(self, s, primitive):
        if primitive == 'reorient':
            box_to_goal = s[5:7] - s[:2] # TODO: Chekc this
            goal_vector = unitVector(box_to_goal)
            goal_direction = math.atan(goal_vector[1]/goal_vector[0])
            curr_direction = s[3]
            return abs(goal_direction - curr_direction) > .2 

    def featurize_state(self, s):
        states = {}         
        states['slope'] = self.primitive_tasks['slope_push'].feature_joint_2_joint(s)
        features = self.primitive_tasks['slope_push'].feature_joint_2_feature(s)
        states['flat'] = self.primmitive_tasks['cross'].feature_2_task_state(features[0]) # assume the second agent leaves after pushing up slope
        return features, states
    
    def split_state(self, s):
        states = {}
        # TODO: Change this to reflect new stuff
        states['slope'] = self.primitive_tasks['slope_push'].feature_joint_2_joint(np.hstack((s[:9], s[10:19], s[20:22])))
        states['feature'] = np.hstack((s[:5], s[6:10]))
        states['feature_next'] = np.hstack((s[:5], s[22:26]))
        states['flat'] = self.primitive_tasks['cross'].feature_2_task_state(states['feature'])
        #states['slope'] = np.hstack((s[:2], s[3:7], s[8:12]))
        #states['flat'] = np.hstack((s[:4], s[10:13]))
        #states['features'] = 
        return states

    def receiveState(self, msg):    
        floats = vrep.simxUnpackFloats(msg.data)
        floats = np.array(floats).ravel()
        phase = floats[-1]
        # Eventually, TODO: featurize the given joint state
        # For each of the agents, determine the next primitive by passing the feature into QDA
        # Then, send the approriate splitted state to the respective task

        # features = self.convertState(np.array(floats[-1]).ravel())
        states = self.split_state(floats[:-1])
        # Eventually, we will use the features to determine next primitive
        a = self.sendActionForPlan(states, phase)  
        return 
    
    def restartProtocol(self, restart):
        if restart == 1:      
            msg = Int8()
            msg.data = 1
            self.curr_rollout = []
            self.fail.publish(msg)

    ######### POST TRAINING #########
    def postTraining(self):
        return 
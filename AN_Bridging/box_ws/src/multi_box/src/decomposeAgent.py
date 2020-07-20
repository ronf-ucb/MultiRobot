#! /usr/bin/env python


import numpy as np 
import torch 
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs. msg import Vector3
from collections import OrderedDict

from Algs.AutoDecompose import Decompose 
from Tasks.decomposeTask import DecomposeTask
from Algs.doubleQ import DoubleQ
from Tasks.hierarchyTask import HierarchyTask
from Tasks.box_slope_task import BoxSlopeTask
from Tasks.omni_box_slope_task import OmniBoxSlopeTask
from Tasks. planner import Planner 
# from Algs.SoftActorCritic import SAC

NAME = 'bot'

algs = {
    0: 'INVERSE',
    1: 'CONTROL',
    2: 'DOUBLE_CONTROL',
    3: 'OMNI_CONTROL',
    4: 'PLANNER'
}
ALGORITHM = 4
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == 'INVERSE':
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot":          {"sub": "/state", "pub": "/action"} #joint action space
            })
    
    params = {
            'clusters':     3,
            'mode':         'RNN', #Spectral, RNN
            'state_n':      4, # this does not include time
            'horizon':      6,
            'noise':        False
        }
    params = {"params": params, "agents": agents}

    bot = Decompose(params, NAME, DecomposeTask())
    
if description == 'CONTROL':
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"sub": "/state", "pub": "/action"} #joint action space
            })

    valPars = {
                'neurons':      (6, 256, 256, 5),
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0, 0, 0, 
                                              0, 0, 0]),
                'std':          torch.Tensor([1, 1, 1,
                                              1, 1, 1]),
                'trainMode':    True,
                'load':         False, 
                'dual':         False,
                }             
    valTrain = {
                'batch':        256, 
                'lr':           3e-4, 
                'buffer':       2000, #500
                'gamma':        .99,
                'explore':      .9, 
                'double':       True,
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = DoubleQ(params, NAME, HierarchyTask())
    #agent = SAC(params, NAME, HierarchyTask())

if description == 'DOUBLE_CONTROL':
    NAME = 'a_bot'
    agents = OrderedDict({
                #ensure ordering matches ros message
                "a_bot": {"sub": "/state", "pub": "/action1"}, #joint action space
            })
    agents["b_bot"] = {"sub": "/state", "pub": "/action2"}
    # agents["c_bot"] = {"sub": "/state", "pub": "/action3"}

    # Placeholders
    valPars = {
                'neurons':      (6, 256, 256, 4),
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0, 0, 0, 0, 0, 0]),
                'std':          torch.Tensor([1, 1, 1, 1, 1, 1]),
                'trainMode':    False,
                'load':         False, 
                'dual':         False,
                }             
    valTrain = {
                'batch':        256, 
                'lr':           3e-4, 
                'buffer':       5000,
                'gamma':        .99,
                'explore':      .9, 
                'double':       True,
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = DoubleQ(params, NAME, BoxSlopeTask())

if description == 'OMNI_CONTROL':
    NAME = 'a_bot'
    agents = OrderedDict({
                #ensure ordering matches ros message
                "a_bot": {"sub": "/state", "pub": "/action1"}, #joint action space
            })
    agents["b_bot"] = {"sub": "/state", "pub": "/action2"}
    agents["c_bot"] = {"sub": "/state", "pub": "/action3"}
    #agents["d_bot"] = {"sub": "/state", "pub": "/action4"}

    # Placeholders
    valPars = {
                'neurons':      (18, 256, 256, 8),
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([-2, 0, 5, 0, 0, 0, 
                                              -1, 0, .5,0, 0, 0,
                                              -1, 0, .5,0, 0, 0]),
                'std':          torch.Tensor([1, 1, 1, np.pi, np.pi, np.pi,
                                              1, 1, 1, np.pi, np.pi, np.pi,
                                              1, 1, 1, np.pi, np.pi, np.pi]),
                'trainMode':    True,
                'load':         False, 
                'dual':         False,
                }             
    valTrain = {
                'batch':        128, 
                'lr':           3e-4, 
                'buffer':       500,
                'gamma':        .99,
                'explore':      .9, 
                'double':       True,
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = DoubleQ(params, NAME, OmniBoxSlopeTask())

if description == 'PLANNER':
    NAME = 'a_bot'
    agents = OrderedDict({
                #ensure ordering matches ros message
                "a_bot": {"sub": "/state", "pub": "/action1"}, #joint action space
            })
    agents["b_bot"] = {"sub": "/state", "pub": "/action2"}
    # agents["c_bot"] = {"sub": "/state", "pub": "/action3"}

    # Placeholders
    valPars = {
                'neurons':      (5, 256, 256, 4),
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0, 0, 0, 0, 0]),
                'std':          torch.Tensor([1, 1, 1, 1, 1]),
                'trainMode':    False,
                'load':         False, 
                'dual':         False,
                } 
    valTrain = {
                'batch':        256, 
                'lr':           3e-4, 
                'buffer':       5000,
                'gamma':        .99,
                'explore':      .9, 
                'double':       True,
                }
    
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    planner = Planner(params, NAME)
while(True):
    x = 1+1
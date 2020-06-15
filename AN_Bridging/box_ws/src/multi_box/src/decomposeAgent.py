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

NAME = 'bot'

algs = {
    0: 'INVERSE',
    1: 'CONTROL'
}
ALGORITHM = 1
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
                'neurons':      (12, 256, 256, 7),
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([-.875, 0, .5, 0, 0, 0, 
                                              -.875, 0, .25,0, 0, 0]),
                'std':          torch.Tensor([1.625, 1.25, .25, np.pi, np.pi, np.pi,
                                              1, 1, .25, np.pi, np.pi, np.pi]),
                'trainMode':    True,
                'load':         False, 
                'dual':         False,
                }             
    valTrain = {
                'batch':        256, 
                'lr':           3e-4, 
                'buffer':       10000,
                'gamma':        .99,
                'explore':      .5, 
                'double':       True,
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = DoubleQ(params, NAME, HierarchyTask())

while(True):
    x = 1+1
#! /usr/bin/env python

import numpy as np 
import torch 
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs. msg import Vector3
from collections import OrderedDict

from Algs.FuN import Feudal 
from Algs.Counterfactual import Counter
from Algs.CounterFeudal import CounterFeudal
from Tasks.boxDoubleTask import BoxDoubleTask

NAME = 'bot'
NAMETWO = 'bot2'

algs = {
    10: "COUNTER",
}
ALGORITHM = 10
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == "COUNTER":
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"sub": "/state", "pub": "/action1"}, #joint action space
                "bot2": {"sub": "/state", "pub": "/action2"}
            })

    actPars = {
                #define hidden state size and input state size...
                'h_state_n':    256,
                'x_state_n':    10, #6 robot, 3 observation, 1: (-1,0,1) for state of rope
                'u_n':          10, # For tanker: 8 movement, 2 front and back interactions # For bridger: 10 movement
                'mu':           torch.Tensor([0 for i in range(10)]),
                'std':          torch.Tensor([1 for i in range(10)]),
                'share_params': False
            }
    actTrain = { 
                'lr':           3e-4, 
                'clip':         1
                }

    valPars = {
                'neurons':      (13, 256, 256, actPars['u_n']), #true state: 6*2 true state, 1: (-1, 0, 1) for state of rope 
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0 for i in range(13)]),
                'std':          torch.Tensor([1 for i in range(13)]),
                'trainMode':    True,
                'load':         False,
                }        
    valTrain = {
                'lr':           3e-4, 
                'w_phase1':     1,
                'w_phase2':     1, 
                'w_phase3':     1,
                'buffer':       10000,
                'gamma':        .99,
                'step':         30,
                }
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, "agents": agents}
    tanker = Counter(params, NAME, BoxDoubleTask())
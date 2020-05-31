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
    11: "COUNTER_FEUDAL"
}
ALGORITHM = 11
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
                'h_state_n':    128,
                'x_state_n':    15, #6 robot, 6 box, 3 observation
                'u_n':          9,
                'mu':           torch.Tensor([0 for i in range(15)]),
                'std':          torch.Tensor([1 for i in range(15)]),
            }
    actTrain = { 
                'lr':           3e-4, 
                'clip':         1
                }

    valPars = {
                'neurons':      (20, 256, 256, actPars['u_n']), #true state: 18, actions other: 1, agent index: 1 
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0 for i in range(20)]),
                'std':          torch.Tensor([1 for i in range(20)]),
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

if description == "COUNTER_FEUDAL":
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"sub": "/state", "pub": "/action1"}, #joint action space
                "bot2": {"sub": "/state", "pub": "/action2"}
            })
    mPars = {
                'neurons':      (18, 256, 256, 1),
                'act':          ['F.leaky_relu', 'F.leaky_relu'],

                # actor parameters
                # latent space
                'width':        256,
                'x_state_n':    18, # true state...calculate advantage function
                'mu':           torch.Tensor([0 for i in range(18)]),
                'std':          torch.Tensor([1 for i in range(18)]),
                'c':            2,
            }
    mTrain = {
                'lr':           3e-4,
                'gamma':        .99
            }
    
    actPars = {
                # hidden state
                'h_state_n':    256,
                'x_state_n':    15, # 6 robot, 6 box, 3 others
                'u_n':          8, # number of actions 
                'k':            16,
                'embedding':    mPars['x_state_n'],
                'mu':           torch.Tensor([0 for i in range(15)]),
                'std':          torch.Tensor([1 for i in range(15)]),
            }
    actTrain = { 
                'lr':           3e-4, 
                'clip':         1
                }

    valPars = { # counterfactual network
                'neurons':      (20, 256, 256, 8), # Input: true_state = 8, actions = 2*1  Output: 8 actions
                'act':          ['F.leaky_relu','F.leaky_relu'],
                'mu':           torch.Tensor([0 for i in range(20)]),
                'std':          torch.Tensor([1 for i in range(20)]),
                'trainMode':    True,
                'load':         False,
                'tau':          .005,
                'int_weight':   .2,
                }        
    valTrain = {
                'lr':           3e-4, 
                'w_phase1':     1,
                'w_phase2':     1, 
                'w_phase3':     1,
                'buffer':       10000,
                'gamma':        .99,
                'step':         8,
                }
    
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, 
            "m_pars": mPars, "m_train": mTrain, "agents": agents}
    tanker = CounterFeudal(params, NAME, BoxDoubleTask())

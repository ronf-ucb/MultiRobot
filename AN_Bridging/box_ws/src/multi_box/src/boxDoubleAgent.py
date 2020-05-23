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
from Algs.AutoFuN import AutoFeudal
from Algs.Counterfactual import Counter
from Tasks.boxDoubleTask import BoxDoubleTask

NAME = 'bot'
NAMETWO = 'bot2'

algs = {
    8: 'FEUDAL',
    9: "AUTO_FEUDAL",
    10: "COUNTER"
}
ALGORITHM = 10
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)


if description == 'FEUDAL':
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"sub": "/state", "pub": "/action1"} ,
                'bot2': {'sub': "/state", 'pub': '/action2'}
            })
    train = {
                'm_gamma':      .99,
                'w_gamma':      .8,
                'lr':           3e-4,
                'w_phase1':     1,
                'w_phase2':     1, 
                'w_phase3':     1,
                'trainMode':    True,
                'clip_grad':    1,
                'step':         40,
                'alpha':        .1,
            }
    fun   = {
                's':            19,
                'u':            64,
                'c':            10,
                'k':            32,
                'd':            256,
            }
    params = {"agents": agents, 'train': train, 'fun': fun}
    agent = Feudal(params, NAME, BoxDoubleTask())

if description == 'AUTO_FEUDAL':
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"sub": "/state", "pub": "/action1"} ,
                'bot2': {'sub': "/state", 'pub': '/action2'}
            })
    train = {
                'm_gamma':      .995,
                'w_gamma':      .8,
                'lr':           3e-4,
                'w_phase1':     1,
                'w_phase2':     1, 
                'w_phase3':     1,
                'trainMode':    True,
                'clip_grad':    1,
                'step':         40,
            }
    fun   = {
                's':            19,
                'u':            8,
                'c':            9,
                'k':            32,
                'd':            256,
                's_reduce':     19,
                's_workers':    13, #assume homogenous state space
                'num_agents':   2
            }
    params = {"agents": agents, 'train': train, 'fun': fun}
    agent = AutoFeudal(params, NAME, BoxDoubleTask())

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
                'u_n':          8,
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
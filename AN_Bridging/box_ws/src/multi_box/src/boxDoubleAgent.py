#! /usr/bin/env python

import numpy as np 
import torch 
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs. msg import Vector3
from collections import OrderedDict

from Algs.QSARSA import CentralQSarsa
from Algs.doubleQ import DoubleQ
from Algs.trpo import TRPOAgent
from Algs.TD3 import Twin_DDPG
from Tasks.boxDoubleTask import BoxDoubleTask


GAMMA = .985
NAME = 'bot'
NAMETWO = 'bot2'

algs = {
    3: "DOUBLE_Q", #REMINDER: if choosing 3, make sure to only run the tankAgent.py in the launch file
    5: "TWIN_DDPG"
}
ALGORITHM = 5
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == "DOUBLE_Q":
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 12, "u": 7 ,"sub": "/state", "pub": "/action1"}, #joint action space
                "bot2": {"n": 6, "u": 7, "sub": "/state", "pub": "/action2"}
            })
    valPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': np.prod([agents[key]["u"] for key in agents.keys()]), #F, FR, FL, B, BL, BR for both
                'hidden': [512,512,512, 512],
                'dual': False,
                'act': [nn.ReLU(),nn.ReLU(),nn.ReLU(), nn.ReLU()],
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': [.10 ,.10 ,.10, .1 ]
                }             
    valTrain = {
                'batch': 16,
                'lr': 1e-7,
                'w_phase1': 100,
                'w_phase2': 250,
                'w_phase3': 300,
                'buffer': 3000,
                'explore': .6, 
                'baseExplore': .1,
                'decay': .7,
                'step': 50,
                'double': True,
                'prioritySample': True,
                'a': 1,
                'l2': .1,
                'gamma': GAMMA
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    bot = CentralQ(params, NAME, BoxDoubleTask("argmax"))

if description == "TWIN_DDPG":
    tau = .05
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 12, "u": 2 ,"sub": "/state", "pub": "/action1"}, #joint action space
                'bot2': {"n": 6, 'u': 2 , 'sub': '/state', 'pub': '/action2'}
            })
    valPars = {
                'in_n': sum([agents[key]["n"] + agents[key]["u"] for key in agents.keys()]),
                'out_n': 1, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': [512, 800, 800, 512, 256],
                'dual': False,
                'tau': tau,
                'act': [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': [.10, .10, .10] 
                }        
    valTrain = {
                'batch': 32,
                'lr': 1e-4,
                'w_phase1': 25,
                'w_phase2': 50,
                'w_phase3': 50,
                'buffer': 3000,
                'explore': (1, .8), #probability and sigma
                'baseExplore': .20,
                'decay': .90,
                'step': 200,
                'prioritySample': True,
                'smooth': .2,
                'clip': .3,
                'policy_delay': 3,
                'a': 1,
                'l2': .01,
                'gamma': GAMMA
                }
    actPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), 
                'hidden': [128, 256, 128], 
                'act': [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                'preprocess': True, 
                'prob': False,
                'dropout': [.1,.1,.1], 
                'loss_fnc': "",
            }
    actTrain = { 
                'lr': 1e-4,
                'l2': .02,
                'gamma': GAMMA
                }
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, "agents": agents}
    bot = Twin_DDPG(params, NAME, BoxDoubleTask("d_policy"))
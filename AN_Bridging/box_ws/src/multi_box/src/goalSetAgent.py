#! /usr/bin/env python

import numpy as np 
import torch 
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs. msg import Vector3
from collections import OrderedDict

from Algs.doubleQ import DoubleQ
from Algs.TD3 import Twin_DDPG
from Algs.A2C import A2C
from Tasks.goalSetTask import GoalSetTask


GAMMA = .985
NAME = 'bot'

algs = {
    3: "DOUBLE_Q", #REMINDER: if choosing 3, make sure to only run the tankAgent.py in the launch file
    5: "TWIN_DDPG",
    6: "A2C"
}
ALGORITHM = 5
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == "DOUBLE_Q":
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 2, "u": 7 ,"sub": "/state", "pub": "/action"}
            })
    valPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), #F, FR, FL, B, BL, BR, S
                'hidden': [256, 256],
                'dual': False,
                'act': [nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': [.2,.2,.2]
                }             
    valTrain = {
                'batch': 16,
                'lr': 1e-6,
                'buffer': 3000,
                'explore': .4, 
                'baseExplore': .05,
                'decay': .8,
                'step': 50,
                'double': True,
                'prioritySample': True,
                'manual': False,
                'a': 1,
                'l2': .1,
                'gamma': GAMMA
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = DoubleQ(params, NAME, GoalSetTask("argmax"))

if description == "TWIN_DDPG":
    tau = .01
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 2, "u": 2 ,"sub": "/state", "pub": "/action"} #joint action space
            })

    valPars = {
                'in_n': sum([agents[key]["n"] + agents[key]["u"] for key in agents.keys()]),
                'out_n': 1, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': [256, 256, 256],
                'dual': False,
                'tau': tau,
                'act': [nn.LeakyReLU(),nn.LeakyReLU(), nn.LeakyReLU()],
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': [.1, .1, .1]
                }        
    valTrain = {
                'batch': 128,
                'lr': 1e-7,
                'lr_decay': (.8, 300),
                'buffer': 3000,
                'explore': .5, #sigma
                'baseExplore': .05,
                'step': 2000,
                'smooth': .1,
                'clip': .2,
                'policy_delay': 2,
                'manual': True,
                'mean': torch.Tensor([0, 0, 0, 0]),
                'variance': torch.Tensor([1.5, 1.5, 3, 3]),
                'a': 1,
                'l2': .01,
                'gamma': GAMMA
                }
    actPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), 
                'hidden': [256, 256, 256], 
                'mean_range': 3,
                'act': [nn.ELU(), nn.ELU(), nn.ELU()],
                'preprocess': True, 
                'prob': False,
                'dropout': [.1, .1, .1], 
                'loss_fnc': "",
            }
    actTrain = { 
                'lr': 1e-8,
                'lr_decay':(.8, 300),
                'l2': .01,
                'gamma': GAMMA,
                'manual': True,
                'mean': torch.Tensor([0, 0]),
                'variance': torch.Tensor([1.5, 1.5])
                }
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, "agents": agents}
    tanker = Twin_DDPG(params, NAME, GoalSetTask("d_policy"))

if description == "A2C":
    tau = .01
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 2, "u": 2 ,"sub": "/state", "pub": "/action"} #joint action space
            })

    valPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': 1, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': [256, 256, 256],
                'dual': False,
                'act': [nn.LeakyReLU(),nn.LeakyReLU(), nn.LeakyReLU()],
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': [0, 0, 0]
                }        
    valTrain = {
                'batch': 8,
                'lr': 1e-6,
                'manual': True,
                'explore': False,
                'mean': torch.Tensor([0, 0]),
                'variance': torch.Tensor([1.5, 1.5]),
                'l2': .01,
                'gamma': GAMMA
                }
    actPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), 
                'hidden': [256, 256, 256], 
                'act': [nn.ELU(), nn.ELU(), nn.ELU()],
                'mean_range': 3,
                'logstd_range': math.log(2),
                'preprocess': True, 
                'prob': True,
                'dropout': [0, 0, 0], 
                'loss_fnc': "",
            }
    actTrain = { 
                'lr': 1e-7,
                'l2': .02,
                'gamma': GAMMA,
                'manual': True,
                'mean': torch.Tensor([0, 0]),
                'variance': torch.Tensor([1.5, 1.5])
                }
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, "agents": agents}
    tanker = A2C(params, NAME, GoalSetTask("p_policy"))


while(True):
    x = 1+1
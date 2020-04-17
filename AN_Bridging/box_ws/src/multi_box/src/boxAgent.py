#! /usr/bin/env python


import numpy as np 
import torch 
import torch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs. msg import Vector3
from agent import Agent
from centralQSARSA import CentralQSarsa
from centralQ import CentralQ
from trpo import TRPOAgent
from collections import OrderedDict
from boxTask import BoxTask
from DDPG import DDPG


GAMMA = .985
NAME = 'bot'

algs = {
    3: "CENTRAL_Q", #REMINDER: if choosing 3, make sure to only run the tankAgent.py in the launch file
    4: "CENTRAL_Q_SARSA", #REMINDER: same as above
    5: "CENTRAL_TRPO",
    6: "CENTRAL_DDPG"
}
ALGORITHM = 3
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == "CENTRAL_Q":
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 12, "u": 6 ,"sub": "/state", "pub": "/action"} #joint action space
            })

    valPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'u_n': sum([agents[key]["u"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': 256,
                'dual': False,
                'depth': 3,
                'act': nn.ReLU(),
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': .10 
                }             
    valTrain = {
                'batch': 16,
                'lr': 1e-5,
                'w_phase1': 100,
                'w_phase2': 100,
                'w_phase3': 100,
                'buffer': 3000,
                'explore': .4, 
                'baseExplore': .1,
                'decay': .6,
                'step': 50,
                'replayDim': 2*valPars['in_n'] + 1 + 1,
                'double': True,
                'prioritySample': True,
                'a': 1,
                'l2': .1,
                'gamma': GAMMA
                }
    params = {"valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = CentralQ(params, NAME, BoxTask("argmax"))

if description == "CENTRAL_Q_SARSA":

    agents = {
                "bot": {"n": 12, "u": 6 ,"sub": "/state", "pub": "/action"}
            }
    valPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'u_n': sum([agents[key]["u"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': 256,
                'dual': False,
                'depth': 3,
                'act': nn.ReLU(),
                'preprocess': True, #makes no sense really
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': .10 
                }             
    valTrain = {
                'batch': 1,
                'lr': 1e-4,
                'w_phase1': 100,
                'w_phase2': 100,
                'w_phase3': 100,
                'buffer': 1,
                'explore': .3, 
                'double' :False,
                'prioritySample': False,
                'a': 0,
                'baseExplore': .1,
                'decay': .75,
                'step': 75,
                'l2': .02,
                'replayDim': 2*valPars['in_n'] + 2 + 1,
                'QWeight': 0,
                'gamma': GAMMA
                }
    params = { "valPars": valPars, "valTrain": valTrain, "agents": agents}
    tanker = CentralQSarsa(params, NAME, BoxTask("argmax"))

if description == "CENTRAL_TRPO":
    agents = {
                #ensure order matches ros messages
                "bot": {"n": 12, "u": 2 ,"sub": "/state", "pub": "/action"}
            }
    actPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), 
                'hidden': 256, 
                'depth': 3, 
                'act': nn.ReLU(),
                'preprocess': True, 
                'prob': True,
                'double' :False,
                'prioritySample': False,
               'dual': False,
                'a': 0,
                'dropout': .1, 
                'loss_fnc': "",
            }
    valPars = {
                'in_n': actPars['in_n'], 
                'u_n': actPars['out_n'],
                'out_n': 1, 
                'hidden': 200, 
                'depth': 3, 
                'act': nn.ReLU(), 
                'preprocess': False, 
                'prob': False, 
                'trainMode': True,
                'double' :False,
                'prioritySample': False,
                'dual': False,
                'a': 0,
                'dropout': .10,
                'loss_fnc': ""
                }
    actTrain = { 
                'explore': 1, 
                'lr': 1e-5,
                'l2': .02,
                'gamma': GAMMA
                }

    valTrain = {
                'batch': 64,
                'lr': 1e-5, 
                'l2': .02,
                'w_phase1': 30,
                'w_phase2': 30,
                'w_phase3': 30,
                'gamma': GAMMA, 
                'explore': 1,
                }
    params = {"actPars": actPars, "valPars": valPars, "actTrain": actTrain, "valTrain": valTrain, "agents": agents}
    tanker = TRPOAgent(params, NAME, BoxTask("p_policy"))

if description == "CENTRAL_DDPG":
    tau = .02
    agents = OrderedDict({
                #ensure ordering matches ros messages
                "bot": {"n": 12, "u": 2 ,"sub": "/state", "pub": "/action"} #joint action space
            })

    valPars = {
                'in_n': sum([agents[key]["n"] + agents[key]["u"] for key in agents.keys()]),
                'u_n': sum([agents[key]["u"] for key in agents.keys()]),
                'out_n': 1, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'hidden': 256,
                'dual': False,
                'depth': 3,
                'tau': tau,
                'act': nn.ReLU(),
                'preprocess': True, 
                'prob': False,
                'trainMode': True,
                'loss_fnc': "MSE",
                'dropout': .10 
                }        
    valTrain = {
                'batch': 32,
                'lr': 1e-5,
                'w_phase1': 25,
                'w_phase2': 100,
                'w_phase3': 50,
                'buffer': 3000,
                'explore': (1, .25), #probability and sigma
                'baseExplore': .15,
                'decay': 0,
                'step': 200,
                'replayDim': 2*valPars['in_n'] - valPars['u_n'] + 1,
                'prioritySample': True,
                'a': 1,
                'l2': .01,
                'gamma': GAMMA
                }
    actPars = {
                'in_n': sum([agents[key]["n"] for key in agents.keys()]),
                'out_n': sum([agents[key]["u"] for key in agents.keys()]), 
                'hidden': 256, 
                'depth': 3, 
                'act': nn.ReLU(),
                'preprocess': True, 
                'prob': False,
                'prioritySample': False,
                'dropout': .1, 
                'loss_fnc': "",
            }
    actTrain = { 
                'lr': 1e-6,
                'l2': .02,
                'gamma': GAMMA
                }
    params = {"valPars": valPars, "valTrain": valTrain, "actPars": actPars, "actTrain": actTrain, "agents": agents}
    tanker = DDPG(params, NAME, BoxTask("d_policy"))

while(True):
    x = 1+1
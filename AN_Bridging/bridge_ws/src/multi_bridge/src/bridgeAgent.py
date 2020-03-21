#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from agent import Agent

GAMMA = .9

actPars = {'state_n': 11, 
            'own_n': 7,
            'output_n': 2,
            'prob': True,
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': False,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "policy_gradient",
            'sigma': 1,
            'dropout': .20
            }

criticPars = {'prob': False,
            'sigma': 1, #relative 
            'prob': False,
            'state_n': 11,
            'output_n': 1,
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': False,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "MSE",
            'dropout': .20
            }
            
actorTrainPars = {'alpha1': 2,
            'alpha2': 2,
            'lambda': .5,
            'horizon': 16,
            'buffer': 1000,
            'explore': 1, #variance of gaussian noise
            'lr': .0000001,
            'gamma': GAMMA
            }
criticTrainPars = {
            'batch': 16,
            'lr': .0000001,
            'gamma': GAMMA
            
}

ROSparams = {'stateSub': "/bridger" ,
                'subQueue': 1,
                'actionPub': "/bridgerSubscribe",
                'pubQueue': 1,
                'delta_t': .05,
                'numAgents': 2
}

bridger = Agent(actPars, criticPars, actorTrainPars, criticTrainPars, ROSparams)
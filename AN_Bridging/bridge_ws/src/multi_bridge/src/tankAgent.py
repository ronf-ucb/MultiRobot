#! /usr/bin/env python

import numpy as np 
import torch 
import torch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String
from geometry_msgs. msg import Vector3
from agent import Agent

GAMMA = .9

actPars = {'state_n': 12,
            'own_n': 7,
            'output_n': 3,
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
criticPars = {
            'sigma': 1, 
            'state_n': 12,
            'output_n': 1,
            'prob': False,
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': False,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "MSE" , 
            'dropout': .20,
            'prob' : False
            }
actorTrainPars = {'alpha1': 2,
            'alpha2': 2,
            'lambda': .5,
            'horizon': 16,
            'buffer': 1000,
            'explore': 1, 
            'lr': .0000001,
            'gamma': GAMMA,
            }

criticTrainPars = {
            'batch': 16,
            'lr': .0000001,
            'gamma': GAMMA
}

ROSparams = {'stateSub': "/tanker",
                'subQueue': 1,
                'actionPub': "/tankerSubscribe",
                'pubQueue': 1,
                'delta_t': .05,
                'numAgents': 2
}


bridger = Agent(actPars, criticPars, actorTrainPars, criticTrainPars, ROSparams)
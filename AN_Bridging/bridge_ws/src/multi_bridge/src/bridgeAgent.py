#! /usr/bin/env python

import numpy as np 
import pytorch as torch
import pytorch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String, Vector3


actPars = {'state_n': 16,
            'output_n': 2,
            'prob': True
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': True,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "policy_gradient",
            'sigma': 1,
            'dropout': .20
            }

criticPars = {'prob': False,
            'sigma': 1, #relative 
            'prob': False
            'state_n': 16,
            'output_n': 1,
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': True,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "MSE",
            'dropout': .20
            }
trainPars {'alpha1': 2,
            'alpha2': 1,
            'lambda': .5,
            'batch': 32,
            'horizon': 16,
            'buffer': 1000,
            'explore': 1, #variance of gaussian noise
            'lr': .0001,
            }

ROSparams = {'stateSub': "bridger" ,
                'subQueue': 1,
                'actionPub': "bridgerSubscribe",
                'pubQueue': 1
}

bridger = agent(actPars, criticPars, actorTrainPars, criticTrainPars, ROSparams)
bridger.train()
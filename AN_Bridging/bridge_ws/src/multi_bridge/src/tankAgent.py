#! /usr/bin/env python

import numpy as np 
import pytorch as torch
import pytorch.nn as nn
import math 
from network import Network
import rospy
from std_msgs.msg import String, Vector3

actPars = {'state_n': 15,
            'output_n': 3,
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
criticPars = {
            'sigma': 1, 
            'state_n': 15,
            'output_n': 1,
            'prob': False,
            'hidden': 100,
            'depth': 2,
            'activation': nn.ReLU(),
            'preprocess': True,
            'postprocess': True,
            'epochs': 1,
            'loss_fnc': "MSE" , 
            'dropout': .20,
            'prob' = False
            }
trainPars {'alpha1': 2,
            'alpha2': 1,
            'lambda': .5,
            'batch': 32,
            'horizon': 16,
            'buffer': 1000,
            'epsilon': 1, #variance of gaussian noise
            'lr': .0001
            }

ROSparams = {'stateSub': "tanker",
                'subQueue': 1,
                'actionPub': "tankerSubscribe",
                'pubQueue': 1
}

bridger = agent(actPars, criticPars, actorTrainPars, criticTrainPars, ROSparams)
bridger.train()
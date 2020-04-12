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
from customAgent import CustomAgent 
from MADDPGAgent import MADDPGAgent 
from centralQSARSA import CentralQSarsa
from centralQ import CentralQ
from trpo import TRPOAgent

GAMMA = .95

algs = {
    1: "CUST_MADDPG_OPT",
    2: "MADDPG",
    3: "CENTRAL_Q", #REMINDER: if choosing 3, make sure to only run the tankAgent.py in the launch file
    4: "CENTRAL_Q_SARSA", #REMINDER: same as above
    5: "CENTRAL_TRPO"
}
ALGORITHM = 3
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description ==  "CUST_MADDPG_OPT":
    actPars = {'state_n': 12,
                'own_n': 7,
                'in_n' : 12,
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
                'in_n': 12,
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
                'gamma': GAMMA}
    ROSparams = {'stateSub': "/tanker",
                'subQueue': 1,
                'actionPub': "/tankerSubscribe",
                'pubQueue': 1,
                'delta_t': .05,
                'numAgents': 2}
    params = {"actorParams": actPars, "valueParams": criticPars, "actorTrain": actorTrainPars, "valueTrain": criticTrainPars, "ROS": ROSparams}
    tanker = CustomAgent(params)
if description == "MADDPG":
    actPars = {'state_n': 12,
                'own_n': 7,
                'in_n': 12,
                'output_n': 3,
                'prob': False,
                'hidden': 100,
                'depth': 2,
                'activation': nn.ReLU(),
                'preprocess': False,
                'postprocess': True,
                'epochs': 1,
                'loss_fnc': "policy_gradient",
                'sigma': 1,
                'dropout': .1
                }
    valuePars = {
                'sigma': 1, 
                'state_n': 12,
                'output_n': 1,
                'in_n': 15,
                'prob': False,
                'hidden': 100,
                'depth': 2,
                'activation': nn.ReLU(),
                'preprocess': False,
                'postprocess': True,
                'epochs': 1,
                'loss_fnc': "MSE" , 
                'dropout': .10,
                }
    actorTrainPars = {'alpha1': 2,
                'alpha2': 2,
                'lambda': .5,
                'horizon': 32,
                'buffer': 1000,
                'explore': 1, 
                'lr': .0000001,
                'gamma': GAMMA,
                }
    valueTrainPars = {
                'batch': 16,
                'lr': .0000001,
                'gamma': GAMMA}
    ROSparams = {'tankSub': "/tanker",
                'subQueue': 1,
                'tankPub': "/tankerSubscribe",
                'pubQueue': 1,
                'delta_t': .05,
                'numAgents': 2}
    params = {"actorParams": actPars, "valueParams": valuePars, "actorTrain": actorTrainPars, "valueTrain": valueTrainPars, "ROS": ROSparams}
    tanker = MADDPGAgent(params)
if description == "CENTRAL_Q":
    valuePars = {'prob': False,
                'sigma': 1, #relative 
                'trainMode': True,
                'output_n': 52, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'u_n': 52,
                'in_n': 14,
                'own_n': 6,
                'other_n': 6,
                'hidden': 500,
                'depth': 3,
                'activation': nn.ReLU(),
                'preprocess': False, #makes no sense really
                'epochs': 1,
                'loss_fnc': "MSE",
                'dropout': .10 }             
    valueTrainPars = {
                'batch': 32,
                'lr': 5e-9,
                'gamma': GAMMA,
                'alpha1': .25,
                'alpha2': 20,
                'alpha3': .4,
                'lambda': 1,
                'buffer': 2000,
                'explore': .85, 
                'baseExplore': .10,
                'decay': .85,
                'step': 200,
                'replayDim': 2*valuePars['in_n'] + 1 + 1}
    ROSparams = {'tankSub': "/tanker" ,
                'subQueue': 1,
                'bridgePub': "/bridgerSubscribe",
                'pubQueue': 1,
                'tankPub': "/tankerSubscribe",
                'delta_t': .05,
                'numAgents': 2}
    params = { "valueParams": valuePars, "valueTrain": valueTrainPars, "ROS": ROSparams}
    bridger = CentralQ(params)
if description == "CENTRAL_Q_SARSA":
    valuePars = {'prob': False,
                'sigma': 1, #relative 
                'trainMode': True,
                'output_n': 52, #F, FR, FL, B, BL, BR, S for both. Else: Unlatch, hook to the front from front, hook to back from back
                'in_n': 10,
                'u_n': 52,
                'own_n': 4,
                'other_n': 4,
                'hidden': 400,
                'depth': 3,
                'activation': nn.ReLU(),
                'preprocess': False, #makes no sense really
                'epochs': 1,
                'loss_fnc': "MSE",
                'dropout': .10 }             
    valueTrainPars = {
                'batch': 8,
                'lr': .000001,
                'gamma': GAMMA,
                'alpha1': .7,
                'alpha2': 5,
                'alpha3': .2,
                'lambda': 1,
                'buffer': 8,
                'explore': .85, 
                'baseExplore': .15,
                'decay': .75,
                'step': 300,
                'replayDim': 2*valuePars['in_n'] + 2 + 1,
                'QWeight': 0}
    ROSparams = {'tankSub': "/tanker" ,
                'subQueue': 1,
                'bridgePub': "/bridgerSubscribe",
                'pubQueue': 1,
                'tankPub': "/tankerSubscribe",
                'delta_t': .05,
                'numAgents': 2}
    params = { "valueParams": valuePars, "valueTrain": valueTrainPars, "ROS": ROSparams}
    bridger = CentralQSarsa(params)
if description == "CENTRAL_TRPO":
    actPars = {'state_n': 10, 'own_n': 4, 'other_n': 4, 'in_n': 10,'output_n': 6, 'hidden': 400, 'depth': 3, 'activation': nn.ReLU(),
            'preprocess': False, 'prob': True,
            'epochs': 1,'sigma': 1, 'dropout': .1, 'loss_fnc': "policy_gradient"}

    valPars = {'state_n': 10, 'own_n':4, 'other_n': 4, 'in_n': 10, 'output_n': 1, 'hidden': 400, 'depth': 3, 'activation': nn.ReLU(), 'u_n': 6,
            'preprocess': False, 'prob': False, 'trainMode': True,
            'epochs': 1, 'sigma': 1, 'dropout': .10,'loss_fnc': "MSE"}

    actTrain = { 'explore': 1, 'lr': .0000001,'gamma': GAMMA}

    valTrain = {'batch': 4,'lr': .0000001, 'gamma': GAMMA, 'alpha1': .7,'alpha2': 5,'alpha3': .2, 'lambda': 1}

    ROSparams = {'tankSub': "/tanker" ,'subQueue': 1,'bridgePub': "/bridgerSubscribe",'pubQueue': 1,'tankPub': "/tankerSubscribe",'delta_t': .05,'numAgents': 2}

    params = {"actorParams": actPars, "valueParams": valPars, "actorTrain": actTrain, "valueTrain": valTrain, "ROS": ROSparams}
    tanker = TRPOAgent(params)
while(True):
    x = 1+1
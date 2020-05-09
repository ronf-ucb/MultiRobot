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
from Algs.FuN import Feudal 
from Algs.AutoFuN import AutoFeudal
from Tasks.boxDoubleTask import BoxDoubleTask

NAME = 'bot'
NAMETWO = 'bot2'

algs = {
    9: "AUTO_FEUDAL"
}
ALGORITHM = 9
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous = True)

if description == 'AUTO_FEUDAL':
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
                'clip_grad':    5,
                'step':         2,
                'alpha':        .1,
            }
    fun   = {
                's':            19,
                'u':            8,
                'c':            9,
                'k':            16,
                'd':            256,
                's_reduce':     8,
                's_workers':    [13, 13],
                'num_agents':   2
            }
    params = {"agents": agents, 'train': train, 'fun': fun}
    agent = AutoFeudal(params, NAME, BoxDoubleTask())

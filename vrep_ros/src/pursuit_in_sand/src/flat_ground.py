#!/usr/bin/env python
# -*- coding: utf-8 -*-

from RL_brain import DeepQNetwork
from test_controller import Robot
import numpy as np
import sys
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
import time
import tensorflow
import vrep

front = "Front"
back = "Back"

# prepare for controllers
robot1 = Robot(front)
robot2 = Robot(back)

class Policy:
    def __init__(self):

        # define DQN algorithm
        tensorflow.reset_default_graph()
        self.RL1 = DeepQNetwork(n_actions=len(robot1.action_space),
                                n_features=len(robot1.observation_space),
                                learning_rate=0.0001, e_greedy=0.9,
                                replace_target_iter=100, memory_size=2000,
                                e_greedy_increment=0.008, )   #0.0008

        self.total_steps = 0
        self.rsrvl = 0.05  # to check
        self.train()

    def train(self):
        vrep.simxFinish(-1) #clean up the previous stuff
        clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if clientID == -1:
            print("Could not connect to server")
            sys.exit()
        
        first = True
        for i_episode in range(100):
            
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

            observation1 = robot1.observation_space
            observation2 = robot2.observation_space
            ep_r = 0
            self.steps = 0
            while True:
                
                action1 = self.RL1.choose_action(observation1)                        # To check
                # print(action1)
                observation1_, done1 = robot1.step(action1)       # To check
                #print(observation1_)
                observation2_ = robot2.observation_space
                done2 = False

                x1, y1, z1, vx1, vy1, vz1, theta1_f, theta2_f, theta3_f = observation1_       # To check
                x2, y2, z2, vx2, vy2, vz2, theta1_b, theta2_b, theta3_b = observation2_  
                
                error, self.r1 = vrep.simxGetObjectHandle(clientID, 'body#1', vrep.simx_opmode_blocking)
                error, self.r2 = vrep.simxGetObjectHandle(clientID, 'body#13', vrep.simx_opmode_blocking)
                error, self.o1 = vrep.simxGetObjectHandle(clientID, 'Obstacle#1', vrep.simx_opmode_blocking)

                error, position_object = vrep.simxGetObjectPosition(clientID, self.o1, -1, vrep.simx_opmode_blocking)
                x1_ = position_object[0]
                y1_ = position_object[1]
                z1_ = position_object[2]
                 
                error, position_hexa_base1 = vrep.simxGetObjectPosition(clientID, self.r1, -1, vrep.simx_opmode_blocking)
                x1 = position_hexa_base1[0]
                y1 = position_hexa_base1[1]
                z1 = position_hexa_base1[2]

                error, position_hexa_base2 = vrep.simxGetObjectPosition(clientID, self.r2, -1, vrep.simx_opmode_blocking)
                x2 = position_hexa_base2[0]
                y2 = position_hexa_base2[1]
                z2 = position_hexa_base2[2]

                distance = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
                distance_object = np.sqrt((x1 - x1_) * (x1 - x1_) + (y1 - y1_) * (y1 - y1_) + (z1 -z1_) * (z1 - z1_))
                ###########################
                if distance>1 or distance<0.15 or distance_object < 0.3:
                    done1 = True
                
                
                ######################################

                #########reward function############
                reward1 = self.rsrvl + (vx1 + vx2) - 0.5 * (np.abs(vy1) + np.abs(vy2))
                reward = 1*(distance < 0.15) - 10*(distance > 1 or distance_object < 0.3) - 0.1*self.steps - 0.1 * distance
                #################################
                #print("R: ", reward)
                print("distance: ", distance)
                print("distance_object",distance_object)
               # print("z1:",z1)

                
                self.RL1.store_transition(observation1, action1, reward, observation1_)

                if self.total_steps > 200 and self.total_steps % 5 ==0:
                    self.RL1.learn()

                ep_r += reward
                if done1 :
                    #print(done1)
                    print('episode: ', i_episode,
                          'ep_r: ', round(ep_r, 2),
                          ' epsilon: ', round(self.RL1.epsilon, 2))
                    break

                observation1 = observation1_
                observation2 = observation2_
                self.total_steps += 1
                self.steps += 1
                done1 = False
        
            first = False
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
            time.sleep(1)
        self.RL1.plot_cost()


if __name__ == "__main__":
    rospy.init_node("cockroachRun")
    Policy()
    rospy.spin()

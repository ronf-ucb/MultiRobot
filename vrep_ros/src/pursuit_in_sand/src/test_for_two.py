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

front = "Front"
back = "Back"

# prepare for controllers
robot1 = Robot(front)
robot2 = Robot(back)

class Policy:
    def __init__(self):
        # define publisher to control start or stop vrep
        self.pub_start_signal = rospy.Publisher("/startSimulation", Bool, queue_size=1)
        self.pub_stop_signal = rospy.Publisher("/stopSimulation", Bool, queue_size=1)

        # maybe start the simulation with hand would be a good way
        time.sleep(2)
        start_signal = Bool()
        start_signal.data = True
        self.pub_start_signal.publish(start_signal)
        time.sleep(2)
        # define DQN algorithm
        tensorflow.reset_default_graph()
        self.RL1 = DeepQNetwork(n_actions=len(robot1.action_space),
                                n_features=len(robot1.observation_space),
                                learning_rate=0.01, e_greedy=0.9,
                                replace_target_iter=100, memory_size=2000,
                                e_greedy_increment=0.0008, )

        self.total_steps = 0
        self.rsrvl = 0.05  # to check
        self.train()

    def train(self):
        for i_episode in range(600):
            stop_signal = Bool()
            stop_signal.data = True
            self.pub_stop_signal.publish(stop_signal)
            time.sleep(0.2)
            start_signal = Bool()
            start_signal.data = True
            self.pub_start_signal.publish(start_signal)

            observation1 = robot1.observation_space
            observation2 = robot2.observation_space
            ep_r = 0
            while True:
                # restart the simulation
                action1 = self.RL1.choose_action(observation1)                        # To check
                # print(action1)
                observation1_, done1 = robot1.step(action1)       # To check
                observation2_, done2 = robot2.step(4)

                x1, y1, z1, vx1, vy1, vz1, theta1_f, theta2_f, theta3_f = observation1_       # To check
                x2, y2, z2, vx2, vy2, vz2, theta1_b, theta2_b, theta3_b = observation2_  
                distance = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

                ###########################
                if distance < 0.07 or z1 < -5 or distance > 1:
                    done1 = True
                ######################################

                #########reward function############
                reward1 = self.rsrvl + (vx1 + vx2) - 0.5 * (np.abs(vy1) + np.abs(vy2))
                reward = reward1 + (distance < 0.03) - 0.5 * np.abs(y2 - y1)
                #################################
                print("R: ", reward)
                print("distance: ", distance)
                print("z1:",z1)

                
                self.RL1.store_transition(observation1, action1, reward, observation1_)

                if self.total_steps > 1000 or done1:
                    self.RL1.learn()  

                ep_r += reward
                if done1 :
                    print(done1)
                    print('episode: ', i_episode,
                          'ep_r: ', round(ep_r, 2),
                          ' epsilon: ', round(self.RL1.epsilon, 2))
                    break

                observation1 = observation1_
                observation2 = observation2_
                self.total_steps += 1
                done1 = False
            stop_ = Bool()
            stop_.data = True
            self.pub_stop_signal.publish(stop_)


if __name__ == "__main__":
    rospy.init_node("cockroachRun")
    Policy()
    rospy.spin()

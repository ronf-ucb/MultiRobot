#!/usr/bin/env python
# -*- coding: utf-8 -*-

from RL_brain import DeepQNetwork
from Controller import Robot
import numpy as np
import sys
import vrep

front = "Front"
back = "Back"

robot1 = Robot(front)  #准备controllers
robot2 = Robot(back)

vrep.simxFinish(-1) #clean up the previous stuff
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    print("Could not connect to server")
    sys.exit()
    

print(robot1.action_space) # 查看这个环境中可用的 action 有多少个
print(robot1.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个


# 定义使用 DQN 的算法
RL1 = DeepQNetwork(n_actions=len(robot1.action_space),
                  n_features=len(robot1.observation_space),
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0008,)

RL2 = DeepQNetwork(n_actions=len(robot2.action_space),
n_features=len(robot2.observation_space),
learning_rate=0.01, e_greedy=0.9,
replace_target_iter=100, memory_size=2000,
e_greedy_increment=0.0008,)


total_steps = 0 # 记录步数
rsrvl = 0.05


for i_episode in range(100):

    #############刷新环境##############
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    ##############刷新环境############
  
    # 获取回合 i_episode 第一个 observation
    observation1 = robot1.observation_space
    observation2 = robot2.observation_space
    
    ep_r = 0
    while True:
        
        action1 = RL1.choose_action(observation1)  # 为机器人1选行为
        action2 = RL2.choose_action(observation2)

        observation1_, done1 = robot1.step(action1) # 获取下一个前一个机器人的 state
        observation2_, done2= robot2.step(action2) # 获取下一个后一个机器人的 state
        
        x1,y1,z1,vx1,vy1,vz1,theta1_f,theta2_f,theta3_f = observation1_   # 前一个机器人的state各参数
        x2,y2,z2,vx2,vy2,vz2,theta1_b,theta2_b,theta3_b = observation2_   # 后一个机器人的state各参数
        distance = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
        
        ########任务1的done的条件###################
        if distance < 0.03:
            done1 = True
            done2 = True
        ########可修改##############################
        
        
        #########平地的reward function############
        reward1 = rsrvl + (vx1 + vx2) - 0.5*(np.abs(vy1)+np.abs(vy2))
        reward = reward1 + (distance < 0.03) - 0.5*np.abs(y2-y1)
        #########可修改########################


        # 保存这一组记忆
        RL1.store_transition(observation1, action1, reward, observation1_)
        RL2.store_transition(observation2, action2, reward, observation2_)
        if total_steps > 1000:
            RL1.learn()  # 学习
            RL2.learn()

        ep_r += reward
        if done1 or done2:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL1.epsilon, 2))
            print('episode: ', i_episode,
            'ep_r: ', round(ep_r, 2),
            ' epsilon: ', round(RL1.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
# 最后输出 cost 曲线
RL1.plot_cost()
RL2.plot_cost()

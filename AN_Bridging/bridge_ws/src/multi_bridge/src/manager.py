#! /usr/bin/env python

import rospy
from std_msgs.msg import Int8
import sys 
import vrep
import time

def receiveStatus(message):
    if message.data == 1: #failure 
        failure = True 
    return

rospy.init_node('Dummy', anonymous = True)
vrep_sub = rospy.Subscriber("/failure", Int8, receiveStatus, queue_size = 1)
episodes = 50 
maxTime = 180 #seconds...3 minutes in this case


if __name__ == "__main__":
    vrep.simxFinish(-1) #clean up the previous stuff
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID == -1:
        print("Could not connect to server")
        sys.exit()
    first = True
    counter = 0
    while (counter < episodes):
        r = 1 
        if not first and r != 0:
            r = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        start = time.time()
        failure = False
        elapsed = 0
        while(not failure and elapsed < maxTime):
            curr = time.time()
            elapsed = curr - start
        vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
        is_running = True
        while is_running:
            error_code, ping_time = vrep.simxGetPingTime(clientID)
            error_code, server_state = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state);
            is_running = server_state & 1
        counter += 1
        first = False

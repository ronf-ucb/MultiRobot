#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
from numpy import linalg
# from follower_pkg.msg import sensor_msg

leaderPub = rospy.Publisher('/vrep_ros_interface/leaderSpeed', Int16MultiArray, queue_size=10)
followerPub = rospy.Publisher('/vrep_ros_interface/followerSpeed', Int16MultiArray, queue_size=10)

# function called after receiving information from the subscriber 
def callback(msg):
    middle_imgIntensity = np.sum(msg.middleSensor[0])
    right_imgIntensity = np.sum(msg.rightSensor[0])
    left_imgIntensity = np.sum(msg.leftSensor[0])
    
    speed = Int16MultiArray(None, [0,0,0,0])
    if right_imgIntensity>left_imgIntensity:
        speed.data = [3,-2,-2,3]
        followerPub.publish(speed)
    elif right_imgIntensity<left_imgIntensity:
        speed.data = [2,-3,-3, 2]
        followerPub.publish(speed)
    else: 
        speed.data = [3,-3,-3,3]
        followerPub.publish(speed)

# Subscriber subscribing to a topic wtih a custom message type contianing the sensor readings
def subscriber():
    rospy.Subscriber("sensorReadings", sensorMsg, callback)

    while not rospy.is_shutdown():
        speed = Int16MultiArray(None, [0,0,0,0])
        if counter < 10:
            speed.data = [3,-3,-3,3]
            leaderPub.publish(speed)
        else:
            speed.data = [3,-2,-2,3]
            leaderPub.publish(speed)
        
        counter += 1
        rospy.sleep(1)


if __name__ == '__main__':

    rospy.init_node('follower', anonymous=True)
    counter = 0
    try:
        subscriber()
    except rospy.ROSInterruptException: pass
    


    # vrep.simxFinish(-1) #clean up the previous stuff
    # clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    # counter = 0
    # with vrepInterface.VRepInterface.open() as vr:
    #   vr.simxStartSimulation(vrep.simx_opmode_oneshot_wait)
    #   try:
    #       while(1):
    #           sensorHandle,sensorHandle_left,sensorHandle_right = getSensorHandles(clientID)
    #           follower_frontMotorHandles = getFollowerMotorHandles(clientID)
    #           leader_frontMotorHandles = getLeaderMotorHandles(clientID)

    #           error, blarg, middle_image = vrep.simxReadVisionSensor(clientID, sensorHandle, vrep.simx_opmode_oneshot_wait)
    #           error, blarg, left_image = vrep.simxReadVisionSensor(clientID, sensorHandle_left, vrep.simx_opmode_oneshot_wait)
    #           error, blarg, right_image = vrep.simxReadVisionSensor(clientID, sensorHandle_right, vrep.simx_opmode_oneshot_wait)

    #           middle_imgIntensity = np.sum(middle_image[0])
    #           right_imgIntensity = np.sum(right_image[0])
    #           left_imgIntensity = np.sum(left_image[0])
    #           if right_imgIntensity>left_imgIntensity:
    #               setMotorSpeeds(clientID, follower_frontMotorHandles, [3,-2,-2,3])
    #           elif right_imgIntensity<left_imgIntensity:
    #               setMotorSpeeds(clientID, follower_frontMotorHandles, [2,-3,-3, 2])
    #           else: 
    #               setMotorSpeeds(clientID, follower_frontMotorHandles, [3,-3,-3,3])

    #           if counter < 10:
    #               setMotorSpeeds(clientID, leader_frontMotorHandles, [3,-3,-3,3])
    #           else:
    #               setMotorSpeeds(clientID, leader_frontMotorHandles, [3,-2,-2,3])
    #           counter += 1
    #   except KeyboardInterrupt: 
    #       print("KeyboardInterrupt: pausing simulation.")
    #       vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)

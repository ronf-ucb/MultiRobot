import vrep
import numpy as np
import matplotlib as plt
import sys
import vrepInterface

def getSensorHandles(clientID):
	error1, sensorHandle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)
	error2, sensorHandle_left = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_left', vrep.simx_opmode_oneshot_wait)
	error3, sensorHandle_right = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_right', vrep.simx_opmode_oneshot_wait)
	if (error1 or error2 or error3):
		print('sensor handle get error')
		vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)
	return [sensorHandle,sensorHandle_left,sensorHandle_right]

def getFollowerMotorHandles(clientID):
	follower_frontMotorHandles = [-1,-1,-1,-1]
	error1, follower_frontMotorHandles[0] = vrep.simxGetObjectHandle(clientID, 'joint_front_left_wheel#0', vrep.simx_opmode_oneshot_wait)
	error2, follower_frontMotorHandles[1] = vrep.simxGetObjectHandle(clientID, 'joint_front_right_wheel#0', vrep.simx_opmode_oneshot_wait)
	error3, follower_frontMotorHandles[2] = vrep.simxGetObjectHandle(clientID, 'joint_back_right_wheel#0', vrep.simx_opmode_oneshot_wait)
	error4, follower_frontMotorHandles[3] = vrep.simxGetObjectHandle(clientID, 'joint_back_left_wheel#0', vrep.simx_opmode_oneshot_wait)
	if (error1 or error2 or error3 or error4):
		print('follower handle get error')
		vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)
	return follower_frontMotorHandles

def getLeaderMotorHandles(clientID):
	leader_frontMotorHandles = [-1,-1,-1,-1]
	error1, leader_frontMotorHandles[0] = vrep.simxGetObjectHandle(clientID, 'leader_joint_front_left_wheel', vrep.simx_opmode_oneshot_wait)
	error2, leader_frontMotorHandles[1] = vrep.simxGetObjectHandle(clientID, 'leader_joint_front_right_wheel', vrep.simx_opmode_oneshot_wait)
	error3, leader_frontMotorHandles[2] = vrep.simxGetObjectHandle(clientID, 'leader_joint_back_right_wheel', vrep.simx_opmode_oneshot_wait)
	error4, leader_frontMotorHandles[3] = vrep.simxGetObjectHandle(clientID, 'leader_joint_back_left_wheel', vrep.simx_opmode_oneshot_wait)
	if (error1 or error2 or error3 or error4):
		print('leader handle get error')
		vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)
	return leader_frontMotorHandles

def setMotorSpeeds(clientID, handles, speeds):
	vrep.simxSetJointTargetVelocity(clientID, handles[0], speeds[0], vrep.simx_opmode_streaming)
	vrep.simxSetJointTargetVelocity(clientID, handles[1], speeds[1], vrep.simx_opmode_streaming)
	vrep.simxSetJointTargetVelocity(clientID, handles[2], speeds[2], vrep.simx_opmode_streaming)
	vrep.simxSetJointTargetVelocity(clientID, handles[3], speeds[3], vrep.simx_opmode_streaming)

if __name__ == "__main__":
	vrep.simxFinish(-1) #clean up the previous stuff
	clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
	counter = 0
	with vrepInterface.VRepInterface.open() as vr:
		vr.simxStartSimulation(vrep.simx_opmode_oneshot_wait)
		try:
			while(1):
				sensorHandle,sensorHandle_left,sensorHandle_right = getSensorHandles(clientID)
				follower_frontMotorHandles = getFollowerMotorHandles(clientID)
				leader_frontMotorHandles = getLeaderMotorHandles(clientID)

				error, blarg, middle_image = vrep.simxReadVisionSensor(clientID, sensorHandle, vrep.simx_opmode_oneshot_wait)
				error, blarg, left_image = vrep.simxReadVisionSensor(clientID, sensorHandle_left, vrep.simx_opmode_oneshot_wait)
				error, blarg, right_image = vrep.simxReadVisionSensor(clientID, sensorHandle_right, vrep.simx_opmode_oneshot_wait)

				middle_imgIntensity = np.sum(middle_image[0])
				right_imgIntensity = np.sum(right_image[0])
				left_imgIntensity = np.sum(left_image[0])
				if right_imgIntensity>left_imgIntensity:
					setMotorSpeeds(clientID, follower_frontMotorHandles, [3,-2,-2,3])
				elif right_imgIntensity<left_imgIntensity:
					setMotorSpeeds(clientID, follower_frontMotorHandles, [2,-3,-3, 2])
				else: 
					setMotorSpeeds(clientID, follower_frontMotorHandles, [3,-3,-3,3])

				if counter < 10:
					setMotorSpeeds(clientID, leader_frontMotorHandles, [3,-3,-3,3])
				else:
					setMotorSpeeds(clientID, leader_frontMotorHandles, [3,-2,-2,3])
				counter += 1
		except KeyboardInterrupt: 
			print("KeyboardInterrupt: pausing simulation.")
			vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)

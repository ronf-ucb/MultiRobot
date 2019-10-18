import vrep
import numpy as np
import matplotlib as plt
import sys
import vrepInterface
from PIL import Image
import array
# import ipdb
# ipdb.set_trace()
CLIENT_ID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

class RoverBot:
	SENSOR_HANDLE_PREFIX = ['Vision_sensor#','Vision_sensor_left#','Vision_sensor_right#']
	MOTOR_HANDLE_PREFIX = ['joint_front_left_wheel#', 'joint_front_right_wheel#','joint_back_right_wheel#','joint_back_left_wheel#']
	NUM_SENSORS = 3
	NUM_MOTORS = 4

	def __init__(self,robot_num):
		self.robot_num = robot_num
		self.defaultSpeed = 3
		self.sensorHandles = self.getSensorHandles()
		self.motorHandles = self.getMotorHandles()
		self.controller = PIDController()

	def getSensorHandles(self):
		sensors = [-1 for i in range(RoverBot.NUM_SENSORS)]
		for i in range(self.NUM_SENSORS):
			errorThrown, sensors[i] = vrep.simxGetObjectHandle(CLIENT_ID, RoverBot.SENSOR_HANDLE_PREFIX[i]+str(self.robot_num), vrep.simx_opmode_oneshot_wait)
			if errorThrown:
				print('sensor handle get error')
				vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)
		return sensors

	def getMotorHandles(self):
		motors = [-1 for i in range(RoverBot.NUM_MOTORS)]
		for i in range(RoverBot.NUM_MOTORS):
			errorThrown, motors[i] = vrep.simxGetObjectHandle(CLIENT_ID, RoverBot.MOTOR_HANDLE_PREFIX[i]+str(self.robot_num), vrep.simx_opmode_oneshot_wait)
			if errorThrown:
				print('motor handle get error')
				vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)
		return motors

	def readSensors(self):
		images = [-1 for i in range(RoverBot.NUM_SENSORS)]
		for i in range(RoverBot.NUM_SENSORS):
			errorThrown,img_shape,image=vrep.simxGetVisionSensorImage(CLIENT_ID,self.sensorHandles[i],1,vrep.simx_opmode_oneshot_wait)
			# errorThrown, resolution, img = vrep.simxGetVisionSensorImage(CLIENT_ID, self.sensorHandles[i], 0, vrep.simx_opmode_oneshot_wait)
			# if errorThrown:
			# 	print('sensor read error')
			images[i] = np.reshape(image, (32,32))
			# print(images[i])
		return images

	def setMotorSpeeds(self, motorInputs = [], controls = False):
		turning_correction,distance_correction = 0,0
		if controls:
			images = self.readSensors()
			self.controller.updateError(images)
			turning_correction,distance_correction = self.controller.runController()

		if len(motorInputs) == 0:
			speed = self.defaultSpeed
			motorInputs = [speed, -speed, -speed, speed]
		motorInputs = [speed+turning_correction for speed in motorInputs]
		# print('dc:',distance_correction)
		motorInputs = [motorInputs[0]+distance_correction,motorInputs[1]-distance_correction, motorInputs[2]-distance_correction, motorInputs[3]+distance_correction]


		for i in range(RoverBot.NUM_MOTORS):
			vrep.simxSetJointTargetVelocity(CLIENT_ID, self.motorHandles[i], motorInputs[i], vrep.simx_opmode_streaming)

class PIDController:
	def __init__(self):
		self.currentError = 0
		self.totalError = 0
		self.previousError = 0
		self.Kp = 1.1/48.
		self.Ki = 0
		self.Kd = .1/48.

		# self.Ki = .1/48.
		# self.Kd = .15/48.

	def runController(self):
		turning_correction = self.Kp*self.currentError+ self.Ki*self.totalError + self.Kd*(self.currentError - self.previousError)
		# distance_correction = -1 if self.distance > .7 else 0
		distance_correction = 0
		return turning_correction, distance_correction


	def updateError(self, images):
		new_error, new_distance = PIDController.getCurrentError(images)
		self.previousError = self.currentError
		self.currentError = new_error
		self.totalError += self.currentError
		self.distance = new_distance/5000.

	def getCurrentError(images):
		middle_image,right_image,left_image = images
		right_image = 1.25*np.array(right_image)
		left_image = 1.25*np.array(left_image)
		combined = np.abs(np.hstack((np.sum(left_image,axis=0),np.sum(middle_image,axis=0),np.flip(np.sum(right_image,axis=0)))))
		# convolved = np.convolve(combined, bump, 'full')
		# print(convolved)
		# print(convolved.shape)
		print(max(combined))
		max_intensity = max(combined)
		idx = [i for i in range(len(combined)) if combined[i] == max_intensity]
		return (-int(np.average(idx) - len(combined)/2),max_intensity)

if __name__ == "__main__":
	vrep.simxFinish(-1) #clean up the previous stuff
	with vrepInterface.VRepInterface.open() as vr:
		vr.simxStartSimulation(vrep.simx_opmode_oneshot_wait)
		try:
			counter = 0
			while(1):
				leader = RoverBot(0)
				follower = RoverBot(1)
				follower1 = RoverBot(2)

				follower.setMotorSpeeds(controls=True)
				# follower1.setMotorSpeeds(controls=True)
				if counter < 10:
					leader.setMotorSpeeds(motorInputs = [4,-4,-4,4], controls=False)
				else:
					leader.setMotorSpeeds(motorInputs = [3,-2.5,-2.5,3],controls=False)
				counter += 1
		except KeyboardInterrupt: 
			print("KeyboardInterrupt: pausing simulation.")
			vr.simxStopSimulation(vrep.simx_opmode_oneshot_wait)

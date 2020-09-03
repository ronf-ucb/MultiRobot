import sim as vrep
import numpy as np
import sys
import cv2
import time
import math

LSignalName = "CycleLeft"
RSignalName = "CycleRight"
BaseFreq = -4

class Pursuit:
    def __init__(self, client, leftName, rightname, vis, followc1, followc2):

        self.go = True
        self.clientID = client
        self.LSignalName = leftName
        self.RSignalName = rightname
        self.visionname = vis

        self.LCycleFreq = BaseFreq
        self.RCycleFreq = BaseFreq

        self.orient = None
        self.kp = 1
        self.kd = -0.3
        self.ki = 0.2
        self.x = None
        self.y = None
        self.prevE = 0
        self.totalE = 0

        self.lower_blue = followc1
        self.upper_blue = followc2

        # for test :: to get the handle
        self.cube = None
        self.cube1 = None
        e, self.cam = vrep.simxGetObjectHandle(clientID, self.visionname, vrep.simx_opmode_blocking)
        e, self.cube = vrep.simxGetObjectHandle(clientID, 'body#7', vrep.simx_opmode_blocking)
        e, self.cube1 = vrep.simxGetObjectHandle(clientID, 'GyroSensor#1', vrep.simx_opmode_blocking)

    def clearSignal(self):
        """
        clear the signal at the very begining
        """
        vrep.simxClearFloatSignal(self.clientID, self.LSignalName, vrep.simx_opmode_oneshot)
        vrep.simxClearFloatSignal(self.clientID, self.RSignalName, vrep.simx_opmode_oneshot)

    def preparation(self):
        """
        implement localization and getPath
        """

    def publish(self):
        """
        send msg to vrep
        msg : the frequency of leg
        """
        vrep.simxSetFloatSignal(self.clientID, self.LSignalName, self.LCycleFreq, vrep.simx_opmode_oneshot)
        vrep.simxSetFloatSignal(self.clientID, self.RSignalName, self.RCycleFreq, vrep.simx_opmode_oneshot)


    def controller(self):
        global counter
        if counter < 500:
            self.RCycleFreq = BaseFreq
            self.LCycleFreq = 0
            print("only right")
            self.publish()
        else:
            self.RCycleFreq = 0
            self.LCycleFreq = BaseFreq
            print("only left")
            self.publish()

    def stop(self, cont):
        if len(cont) == 0:
            self.LCycleFreq = 0
            self.RCycleFreq = 0
            self.publish()
            print(result)
            self.go = False
            return False
        else:
            return True

# vrep stuff
vrep.simxFinish(-1)  # clean up the previous stuff
clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection unsuccessful')
    sys.exit('Error: Could not connect to API server')


pursuit = Pursuit(clientID, LSignalName, RSignalName, 'Vision_sensor', np.array([40, 40, 40]), np.array([70, 255, 255]))
error, res, i = vrep.simxGetVisionSensorImage(clientID, pursuit.cam, 0, vrep.simx_opmode_streaming)

time.sleep(2)
counter = 0


pursuit.clearSignal()
while True:
    counter+=1
    pursuit.preparation()
    pursuit.controller()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

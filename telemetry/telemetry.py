# Using pd controller to kamigami as the path tracking method.


import vrep
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import math
import csv
import datetime
import keyboard

class Kamigami:
    def __init__(self, client, BaseFreq = 3):
        self.clientID = client
        self.LSignalName = "CycleLeft"
        self.RSignalName = "CycleRight"

        self.BaseFreq = BaseFreq
        self.LCycleFreq, self.RCycleFreq = BaseFreq, BaseFreq

        self.forceSensors = dict()
        self.positionSensors = dict()
        self.telemetryData = dict()


    def clearSignal(self):
        """
        clear the signal at the very begining
        """
        vrep.simxClearFloatSignal(self.clientID, self.LSignalName, vrep.simx_opmode_oneshot)
        vrep.simxClearFloatSignal(self.clientID, self.RSignalName, vrep.simx_opmode_oneshot)

    def getSensorHandles(self):
        """
        implement localization and getPath
        """
        error, self.positionSensors["body"] = vrep.simxGetObjectHandle(clientID, 'body#1', vrep.simx_opmode_blocking)
        error, self.positionSensors["gyro"] = vrep.simxGetObjectHandle(clientID, 'GyroSensor#1', vrep.simx_opmode_blocking)
        error, self.forceSensors['body'] = vrep.simxGetObjectHandle(clientID, 'Body_force_sensor', vrep.simx_opmode_blocking)
        error, self.forceSensors['L1'] = vrep.simxGetObjectHandle(clientID, 'L1_force_sensor', vrep.simx_opmode_blocking)
        error, self.forceSensors['R1'] = vrep.simxGetObjectHandle(clientID, 'R1_force_sensor', vrep.simx_opmode_blocking)


    def controller(self):
        self.LCycleFreq = self.BaseFreq
        self.RCycleFreq = self.BaseFreq
        error = vrep.simxSetFloatSignal(self.clientID, self.LSignalName, self.LCycleFreq, vrep.simx_opmode_oneshot)
        error = error or vrep.simxSetFloatSignal(self.clientID, self.RSignalName, self.RCycleFreq, vrep.simx_opmode_oneshot)
        if error != 0:
            print("Function error: ", error)

        time=vrep.simxGetLastCmdTime(clientID)
        self.telemetryData['cycleFrequencyTime'] = time
        self.telemetryData['leftCycleFrequency'] = self.LCycleFreq
        self.telemetryData['rightCycleFrequency'] = self.RCycleFreq


    def readForceSensors(self):
        for sensor in self.forceSensors.keys():
            error, state, forceVector, torqueVector = vrep.simxReadForceSensor(self.clientID, self.forceSensors[sensor], vrep.simx_opmode_blocking)
            time=vrep.simxGetLastCmdTime(clientID)
            if error == 0: 
                self.XYZdata(sensor+'force', forceVector, time)
                self.XYZdata(sensor+'torque', torqueVector, time)
            else:
                print("Force sensor read error: %d", error)

    def readPositionSensors(self):
        """
        get the position of robot   TODO
        """
        # get position
        error, position_hexa_base = vrep.simxGetObjectPosition(clientID, self.positionSensors["body"], -1,
                                                                        vrep.simx_opmode_blocking)
        time=vrep.simxGetLastCmdTime(clientID)
        if error == 0: 
            self.XYZdata('bodyPosition', position_hexa_base, time)
        else:
            print("Position sensor read error: %d", error)

        # get orientation
        error, orientation_hexa_base = vrep.simxGetObjectOrientation(clientID, self.positionSensors["body"], -1,
                                                                              vrep.simx_opmode_blocking)
        time=vrep.simxGetLastCmdTime(clientID)
        if error == 0: 
            self.XYZdata('bodyOrientation', orientation_hexa_base, time)
        else:
            print("Orientation sensor read error: %d", error)

        ############################################################################

        error, position_hexa = vrep.simxGetObjectPosition(clientID, self.positionSensors["gyro"], -1,
                                                                        vrep.simx_opmode_blocking)
        time=vrep.simxGetLastCmdTime(clientID)
        if error == 0: 
            self.XYZdata('gyro', position_hexa, time)
        else:
            print("Gyro sensor read error: %d", error)

    def XYZdata(self, name, data, time):
        self.telemetryData[name+'Time'] = time
        self.telemetryData[name+'X'] = data[0]
        self.telemetryData[name+'Y'] = data[1]
        self.telemetryData[name+'Z'] = data[2]


    def doTelemetry(self):
        self.readPositionSensors()
        self.readForceSensors()
        return self.telemetryData

vrep.simxFinish(-1)  # clean up the previous stuff
clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection unsuccessful')
    sys.exit('Error: Could not connect to API server')


kamigami = Kamigami(clientID)

loop = True
forceVectors = np.array([0,0,0])
kamigami.getSensorHandles()

dt = datetime.datetime.now()
filename = './telemetry/'+'cycleFreq3_'+str(dt.month)+'-'+str(dt.day)+'_'+str(dt.hour)+str(dt.minute)+'.csv'
print(filename)

with open(filename,'w' , newline='') as csvfile:
    fieldnames = ['bodyPositionTime', 'bodyPositionX', 'bodyPositionY', 'bodyPositionZ']
    fieldnames += ['bodyOrientationTime','bodyOrientationX', 'bodyOrientationY', 'bodyOrientationZ']
    fieldnames += ['gyroTime','gyroX', 'gyroY', 'gyroZ'] 
    fieldnames += ['cycleFrequencyTime', 'leftCycleFrequency','rightCycleFrequency'] 
    fieldnames += ['bodyforceTime', 'bodyforceX', 'bodyforceY', 'bodyforceZ', 'bodytorqueTime', 'bodytorqueX', 'bodytorqueY', 'bodytorqueZ'] 
    fieldnames += ['R1forceTime', 'R1forceX', 'R1forceY', 'R1forceZ', 'R1torqueTime', 'R1torqueX', 'R1torqueY', 'R1torqueZ']
    fieldnames += ['L1forceTime', 'L1forceX', 'L1forceY', 'L1forceZ', 'L1torqueTime', 'L1torqueX', 'L1torqueY', 'L1torqueZ']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    while loop:     
        kamigami.clearSignal()
        kamigami.controller()
        writer.writerow(kamigami.doTelemetry())
        time.sleep(0.1)

        if keyboard.is_pressed('y'):
             break


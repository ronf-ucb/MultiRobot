#! /usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Float32
from DStarNavigator.msg import all_parameters
import matplotlib.pyplot as plt
import sys
import heapq
import math
from AN_Environment import Environment
import operator


'''Angles returned: [0] positive means angle inclination. Negative indicates angle decline
                    [1] amount of twist. Looking at robot behind to front: Positive means clockwise, negative counter
                    [2] angle from bottom of x axis of world frame '''
'''Proximity Sensor Vector (Normal unit vector to surface detected):
                    [0] positive is left of robot
                    [1] positive is up from robot
                    [2] positive is away from robot'''

WALLBACK = 150000
SLOPETRAVEL = 1
MINDISTANCE = .02
PADDING = 1 #degrees of padding. Specifies what range of qualifies as "'going in the desired direction'"
RESOLUTION = 8
MAPDIMENSIONS = (100, 100 ,2) #g, rhs, slopes, elevation (slopes and elevation not included)
CLIFFTHRESHOLD = .01 #height of a cliff to be considered dangerous

class DStarAgent(object):
    def __init__(self):
        #ROS
        rospy.Subscriber("/robPos", Float32MultiArray, self.rosRobPos, queue_size = 1)
        rospy.Subscriber("/goalPos", Float32MultiArray, self.rosGoalPos, queue_size = 1)
        rospy.Subscriber("/robOrient", Float32MultiArray, self.rosRobOrient, queue_size = 1)
        rospy.Subscriber("/proxDist", Float32, self.rosProxDist, queue_size = 1)
        rospy.Subscriber("/proxVec", Float32MultiArray, self.rosProxVec, queue_size = 1)
        rospy.Subscriber("/3DSense", Float32MultiArray, self.ros3DSensor, queue_size = 1)

        self.keyFactor = 0
        self.open = list()
        self.env = Environment(RESOLUTION, MAPDIMENSIONS, CLIFFTHRESHOLD)
        self.currHeight = 0
        self.orientation = None
        self.distance = None 
        self.proxVec = None 
        self.data3D = None 
        self.dim = None 

    def policy(self):
        ########## WRAPPER FOR GENERAL AGENT POLICY ##################
        while (not self.env.goalDone()):
            print("RECOMPUTING SHORTEST PATH")
            self.computeShortestPath() #update the map
            self.env.showColorGrid()
            print("UPDATING START")
            ########### CONTINUE UNTIL DETECT NEW OBSTACLE ##############
            while(True):
                ########### CHECK FOR OBSTACLE ##############
                pos = self.env.robotPosition
                cliffs = self.checkGround(pos)
                if pos in self.env.cliffs.keys():
                    cliffs = cliffs - self.env.cliffs[pos] #difference
                if len(cliffs) > 0:
                    print("NEW CLIFF DETECTED")
                    self.stopRobot()
                    self.manageCliff(pos, cliffs)
                    break
                obstacleAndLocation = self.checkProximity()
                if obstacleAndLocation[0]:
                    print("NEW OBSTACLE DETECTED")
                    self.stopRobot()
                    self.manageObstacle(obstacleAndLocation[1])
                    break
                ########## CHOOSE OPTIMAL POSITION TO TRAVEL TO #############
                neighbors = self.env.neighbors(pos)
                costs = []
                dots = []
                angle = self.orientation
                angle = angle[2]
                v = (math.cos(angle), math.sin(angle))
                for n in neighbors:
                    costs += [self.env.edge(pos, n) + self.env.map[n[0], n[1], 0]]
                    dots += [self.env.dotProduct((n[0] - pos[0], n[1] - pos[1]),v)]

                ############## UPDATE POSITION #############
                if costs[dots.index(max(dots))] == np.inf and neighbors[dots.index(max(dots))] not in self.env.obstacles:
                    #if the direction we are facing has infinite value and is a cliff
                    self.updateStart((0,0), "back")
                #END CHANGE
                else:
                    minimum = min(costs)
                    indices = [i for i,x in enumerate(costs) if x == minimum]
                    if len(indices) == 0:
                        self.updateStart(neighbors[indices[0]])
                    else:
                        dots = [dots[i] for i in indices]
                        candidates = [neighbors[i] for i in indices]
                        minPoint = candidates[dots.index(max(dots))]
                        self.updateStart(minPoint)

    def manageObstacle(self, location):
        ######### DETECTED OBJECT. REMOVE FROM PQ. UPDATE COSTS OF NEIGHBORS ###########
        self.env.map[location[0], location[1], 0] = np.inf 
        self.env.map[location[0], location[1], 1] = np.inf 
        inQueue = [entry for entry in self.open if entry[1] == location]
        for e in inQueue:
            self.open.remove(e)
        neighbors = self.env.neighbors(location)
        for n in neighbors:
            if n not in self.env.obstacles:
                self.updateState(n)

    def manageCliff(self, robotPosition, cliffs):
        self.env.cliffs[robotPosition] = self.env.cliffs[robotPosition].union(cliffs) if robotPosition in self.env.cliffs else cliffs #update the set of cliffs
        update = []
        for vector in cliffs:
            update += [tuple(map(operator.add, robotPosition, vector))]
        for newCliff in update:
            self.updateState(newCliff)

    def updateStart(self, newPosition, default = None): #pass in default to just take a certain action
        while(True):
            ############ GET INITIAL POSITIONS/ANGLES. CHECK IF TOO CLOSE TO OBSTACLE ############
            position = self.env.robotPosition
            angle = self.orientation
            distance = self.distance
            if distance != -1 and distance < MINDISTANCE:
                self.backRobot()
                position = self.env.robotPosition
                break
            else:
                ########## TRAVEL TO THE NEW POSITION ##############
                if not default:
                    angle = self.env.radToDeg(angle[2]) #the angles that v-rep gives range from -pi to pi radians. This is hard to work with. Convert to 0 to 360 degrees
                    angle = angle + 360 if angle < 0 else angle
                    xVec = newPosition[0] - position[0]
                    yVec = newPosition[1] - position[1]
                    desiredAngle = math.degrees(math.atan(yVec/xVec)) if xVec != 0 else yVec * 90
                    desiredAngle = desiredAngle + 360 if desiredAngle < 0 else desiredAngle

                    if desiredAngle - PADDING < angle and desiredAngle + PADDING > angle:
                        self.goStraight()
                    else:
                        turnRight = ((360 - desiredAngle) + angle) % 360
                        turnLeft = ((360 - angle) + desiredAngle) % 360
                        if turnRight < turnLeft: #turn right if the work to turn right is less than turning left
                            self.turnRight()
                        else: #turn left if the work to turn left is less than turning right
                            self.turnLeft()
                    self.sendSignal()
                else:
                    if default == "back":
                        self.backRobot()
                    else:
                        print("Error: not implemented")
                        sys.exit()

            ######### BREAK AND UPDATE ROBOTPOSITION IF TRANSITIONED ###########
            if position != self.env.robotPosition:
                self.env.robotPosition = position
                difference = self.env.euclidian(position, self.env.robotPosition) #the difference in heuristic is this anyway
                self.keyFactor += difference
                break

    def updateState(self, s):
        ######## UPDATE THIS STATE BY CALCULATING NEW RHS VALUE ##########
        if s != self.env.goalPosition:
            minimum = np.inf
            gPlusEdge = []
            for n in self.env.neighbors(s):
                gPlusEdge += [self.env.map[n[0], n[1], 0] + self.env.edge(s, n)]
            minimum = min(gPlusEdge)
            self.env.map[s[0], s[1], 1] = minimum
        inQueue = [entry for entry in self.open if entry[1] == s]

        ######### REMOVE FROM QUEUE IF PRESENT ##########
        if len(inQueue) > 0:
            self.open.remove(inQueue[0])
        ######## ADD BACK IN WITH UPDATED VALUES IF INCONSISTENT ##########
        if self.env.map[s[0], s[1], 0]  != self.env.map[s[0], s[1], 1]:
            heapq.heappush(self.open, (self.key(s), s))

    def computeShortestPath(self):
        ######## PROPAGATE ALL CHANGES TO FIND SHORTEST PATH ############
        while (len(self.open) > 0):
            node = heapq.heappop(self.open) #the points are already transformed
            currPoint = node[1]
            x = currPoint[0]
            y = currPoint[1]
            g = self.env.map[x, y, 0]
            rhs = self.env.map[x, y, 1]
            neighbors = self.env.neighbors(currPoint)

            ######## CHECK FOR CONSISTENCY, UNDERCONSISTENCY, AND OVERCONSISTENCY ##########
            if g == rhs:
                continue
            if g > rhs:
                self.env.map[x,y,0] = rhs
            else:
                self.env.map[x,y,0] = np.inf
                n = n + [currPoint]
            for n in neighbors:
                if n not in self.env.obstacles:
                    self.updateState(n)

    def checkGround(self, robotPosition):
        table = self.data3D
        if self.dim*self.dim*3 != len(table):
            print("Error with 3D data size")
            return set()
        heights = np.array(table).reshape((self.dim, self.dim, 3))[:,:,0]
        cliffs = self.env.analyzeCliffs(heights) #returns a set of relative locations of cliffs
        return cliffs

    def checkProximity(self):
        ###### DETECT NEW OBSTACLE AND ADD TO OUR ARRAY SPACE GRAPH ############
        if self.distance == -1: #if nothing was detected '''TODO: work on defaults here'''
            return (False, None)
        ##### IF SOMETHING WAS DETECTED ##########
        self.stopRobot()
        distance = self.distance
        vector = self.proxVec
        angle = self.orientation
        currPosition = self.env.inverseTransform(self.env.robotPosition)
        vector = self.env.rotate(vector, angle)
        slope = -((vector[0]**2 + vector[1]**2)**(1/2))/vector[2] if vector[1] > 0 else ((vector[0]**2 + vector[1]**2)**(1/2))/vector[2]

        if slope < SLOPETRAVEL: #if the detection has a slope considered travelable
            return (False, None)

        xdist = math.cos(angle[2]) * distance
        ydist = math.sin(angle[2]) * distance
        location = self.env.transform((xdist + currPosition[0], ydist + currPosition[1]))
        self.env.map[location[0], location[1], 2] = np.inf 

        ####### IF IT IS A NEW OBSTACLE, RETURN THE RESULTS #########
        if location in self.env.obstacles:
            return (False, None)
        return (True, location)

    def prepare(self):
        ################ GET ROBOT/GOAL POSITIONS, MAP DIM WITH RESOLUTION, TRANSFORM ###########################
        while (self.env.robotPosition == None):
            x = 1 + 1 #something random
        while (self.env.goalPosition == None):
            x = 1 + 1
        while (self.data3D == None):
            x = 1 + 1
        self.dim = int((len(self.data3D) / 3) ** (1/2))
        self.env.initializeMap()

        ############# INITIALIZE PRIORITY QUEUE ######################
        heapq.heapify(self.open)
        goalPosition = self.env.goalPosition
        self.env.map[goalPosition[0], goalPosition[1], 1] = 0
        heapq.heappush(self.open, (self.key(goalPosition), goalPosition))

    def backRobot(self):
        for i in range(WALLBACK):
            self.goBack()
        self.stopRobot()

    def key(self, point):
        x = point[0]
        y = point[1]
        cost = min(self.env.map[x,y,0], self.env.map[x,y,1])
        return (cost + self.calcHeuristic(point) + self.keyFactor, cost)

    def calcHeuristic(self, point):
        #calculates the heuristic of a given point. Heuristic should equal the distance from start plus key factor
        return (self.env.euclidian(point, self.env.robotPosition))

    def rosRobPos(self, message):
        self.env.robotPosition = self.env.transform(message.data)

    def rosGoalPos(self, message):
        self.env.goalPosition = self.env.transform(message.data)

    def rosRobOrient(self, message):
        self.orientation = message.data 

    def rosProxDist(self, message):
        self.distance = message.data

    def rosProxVec(self, message):
        self.proxVec = message.data

    def ros3DSensor(self, message):
        self.data3D = message.data

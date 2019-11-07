import rospy
from std_msgs.msg import Int16
from AN_DStarAgent import DStarAgent

LSignalName = "CycleLeft"
RSignalName = "CycleRight"

class LeggedDStar(DStarAgent):
    def __init__(self, leftName, rightName):
        super(LeggedDStar, self).__init__() #call parent class init
        #ROS: publish to V-Rep
        self.pubRight = rospy.Publisher("/rightFrequency", Int16, queue_size = 1)
        self.pubLeft = rospy.Publisher("/leftFrequency", Int16, queue_size = 1)
        #robot movement
        self.LSignalName = leftName
        self.RSignalName = rightName
        self.CycleFreqL = 1
        self.CycleFreqR = 1

    #ALL DRIVING FUNCTIONS ARE CALLED DIRECTLY FROM THE AGENT. NO NEED TO IMPLEMENT ROS FUNCTINALITY
    def goStraight(self):
        self.CycleFreqL = 4
        self.CycleFreqR = 4
        self.sendSignal()

    def turnRight(self):
        self.CycleFreqL = 4
        self.CycleFreqR = 2
        self.sendSignal()

    def turnLeft(self):
        self.CycleFreqL = 2
        self.CycleFreqR = 4
        self.sendSignal()

    def goBack(self):
        self.CycleFreqL = -4
        self.CycleFreqR = -4
        self.sendSignal()

    def stopRobot(self):
        self.CycleFreqL = 0
        self.CycleFreqR = 0
        self.sendSignal()

    def sendSignal(self):
        self.pubRight.publish(self.CycleFreqR)
        self.pubLeft.publish(self.CycleFreqL)

if __name__ == "__main__":
    rospy.init_node("RobotSignals", anonymous=True)
    agent = LeggedDstar(LSignalName, RSignalName)
    agent.prepare()
    agent.policy()

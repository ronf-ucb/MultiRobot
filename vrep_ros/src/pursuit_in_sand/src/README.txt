This version has two launch.


###################
coop_climb.launch:#
###################
Trying to do:
We are trying to let the first robot follow the line on a bevel face and than climb on a dune. If the first robot stuck in the dune and cannot move forward, the second robot will go and push the first robot.

Difficulty:
1. This week we use compare the moving distance of the robot to judge if the robot stucks in the dune. But this judgement seems not stable.
2. The robot will slide down from the dune. Then the orientation of the robot may change. So the second robot may push the first robot to a wrong direction.
3. As there is no link between two robots, the second robot sometimes fail to push the first robot.
4. The pursuit performance on a slope is not good...


TODO : 
We are trying to use pitch angle from gyro sensor to judge it. 
When the robot perceives the the dune using gyro sensor, it will stop and wait the second robot.


###################
link_coop.launch: #
###################
Trying to do:
Angela has completed a model of linked robots. We tried to use this model to help the robot to climb a dune.

Difficulty:
The linked robots have a better performance than seperate robot. But it is not stable as well. 
1. The first robot may slide down from the dune.
2. The second robot may push the first robot to a wrong direction.


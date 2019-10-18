# Install script for directory: /home/vrep/MultiRobot/ros_workspace/follow2/src/vrep_skeleton_msg_and_srv

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/vrep/MultiRobot/ros_workspace/follow2/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vrep_skeleton_msg_and_srv/srv" TYPE FILE FILES "/home/vrep/MultiRobot/ros_workspace/follow2/src/vrep_skeleton_msg_and_srv/srv/displayText.srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vrep_skeleton_msg_and_srv/cmake" TYPE FILE FILES "/home/vrep/MultiRobot/ros_workspace/follow2/build/vrep_skeleton_msg_and_srv/catkin_generated/installspace/vrep_skeleton_msg_and_srv-msg-paths.cmake")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/vrep/MultiRobot/ros_workspace/follow2/devel/include/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/vrep/MultiRobot/ros_workspace/follow2/devel/share/roseus/ros/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/vrep/MultiRobot/ros_workspace/follow2/devel/share/common-lisp/ros/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/vrep/MultiRobot/ros_workspace/follow2/devel/share/gennodejs/ros/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/vrep/MultiRobot/ros_workspace/follow2/devel/lib/python2.7/dist-packages/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/vrep/MultiRobot/ros_workspace/follow2/devel/lib/python2.7/dist-packages/vrep_skeleton_msg_and_srv")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/vrep/MultiRobot/ros_workspace/follow2/build/vrep_skeleton_msg_and_srv/catkin_generated/installspace/vrep_skeleton_msg_and_srv.pc")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vrep_skeleton_msg_and_srv/cmake" TYPE FILE FILES "/home/vrep/MultiRobot/ros_workspace/follow2/build/vrep_skeleton_msg_and_srv/catkin_generated/installspace/vrep_skeleton_msg_and_srv-msg-extras.cmake")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vrep_skeleton_msg_and_srv/cmake" TYPE FILE FILES
    "/home/vrep/MultiRobot/ros_workspace/follow2/build/vrep_skeleton_msg_and_srv/catkin_generated/installspace/vrep_skeleton_msg_and_srvConfig.cmake"
    "/home/vrep/MultiRobot/ros_workspace/follow2/build/vrep_skeleton_msg_and_srv/catkin_generated/installspace/vrep_skeleton_msg_and_srvConfig-version.cmake"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/vrep_skeleton_msg_and_srv" TYPE FILE FILES "/home/vrep/MultiRobot/ros_workspace/follow2/src/vrep_skeleton_msg_and_srv/package.xml")
endif()


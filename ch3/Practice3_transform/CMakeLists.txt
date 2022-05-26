cmake_minimum_required( VERSION 2.8 )
project(example)

include_directories("usr/include/eigen3")
add_executable(TwoRobotTransform TwoRobotTransform.cpp)

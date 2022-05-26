cmake_minimum_required(VERSION 2.8)

project(Eigenbasic)

set(CMAKE_BUILD_TYPE "Release")
# 调节编译时的优化程度， max is O3, min is O0(None)
set(CMAKE_CXX_FLAGS "-O3")

include_directories("usr/include/eigen3")
add_executable(eigenbasic eigenMatrix.cpp)
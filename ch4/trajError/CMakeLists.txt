find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})
target_link_libraries(trajectoryError Sophus::Sophus)

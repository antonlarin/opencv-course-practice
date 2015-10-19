cmake_minimum_required(VERSION 2.8)
project(techvis-adaptive-filter)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(af main.cpp)
target_link_libraries(af ${OpenCV_LIBS})


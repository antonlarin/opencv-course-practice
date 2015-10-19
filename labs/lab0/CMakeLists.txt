cmake_minimum_required(VERSION 2.8)
project(OpenCVTest)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(ocvtest main.cpp)
target_link_libraries(ocvtest ${OpenCV_LIBS})


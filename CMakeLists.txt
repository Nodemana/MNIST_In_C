cmake_minimum_required(VERSION 3.10)
project(MNIST_C_NN)

set(CMAKE_C_STANDARD 11)

find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS}) ${OpenCV_LIBS}

add_executable(cmnist main.c matrixmath.c neural.c sigmoid.c)
target_link_libraries(cmnist m)

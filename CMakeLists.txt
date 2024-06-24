cmake_minimum_required(VERSION 3.10)
project(MNIST_C_NN)

set(CMAKE_C_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cmnist main.c)
target_link_libraries(cmnist ${OpenCV_LIBS})

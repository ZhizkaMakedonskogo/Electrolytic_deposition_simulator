cmake_minimum_required(VERSION 3.16)
project(inverse_problem_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(inverse_problem src/main.cpp)


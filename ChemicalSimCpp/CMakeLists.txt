cmake_minimum_required(VERSION 3.16)
project(ChemicalSimCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(chemical_sim_cli
    src/main.cpp
    src/cluster.cpp
    src/board.cpp
    src/metrics.cpp
    src/simulation_objective.cpp
)

target_include_directories(chemical_sim_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    target_compile_options(chemical_sim_cli PRIVATE /W4 /permissive-)
else()
    target_compile_options(chemical_sim_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

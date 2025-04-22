cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp14x050)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(exercise_2 exercise_2.cpp)
target_link_libraries(exercise_2 "${TORCH_LIBRARIES}")
set_property(TARGET exercise_2 PROPERTY CXX_STANDARD 17)
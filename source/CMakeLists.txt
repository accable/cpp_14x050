cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp14x050)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(exercise_4 exercise_4.cpp)
target_link_libraries(exercise_4 "${TORCH_LIBRARIES}")
set_property(TARGET exercise_4  PROPERTY CXX_STANDARD 17)

if(APPLE)
    # Use Accelerate Framework (Apple's optimized BLAS)
    set(BLA_VENDOR Apple)
    
    # Enable Metal acceleration if available
    if(EXISTS "/System/Library/Frameworks/Metal.framework")
        add_definitions(-DUSE_METAL)
        message(STATUS "Metal framework found - enabling GPU acceleration")
    endif()
    
    # Additional Clang optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvectorize -fslp-vectorize")
endif()
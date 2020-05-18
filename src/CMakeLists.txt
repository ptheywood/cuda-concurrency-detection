# Cmake configuraiton stuff
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Get the repo root
get_filename_component(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Add custom modules directory
set(CMAKE_MODULE_PATH ${REPO_ROOT}/cmake/modules/ ${CMAKE_MODULE_PATH})

set(EXECUTABLE_NAME "cuda-concurrency-detection")

# Include common rules
include(${REPO_ROOT}/cmake/cuda-gencodes.cmake)
include(${REPO_ROOT}/cmake/compiler-flags.cmake)


# Set Include files (.h, .hxx, .cuh)
set(INCLUDE_ROOT ${REPO_ROOT}/include)
set(INCLUDE_FILES
    ${INCLUDE_ROOT}/util/ComputeCapability.cuh
    ${INCLUDE_ROOT}/util/CUDADevice.cuh
    ${INCLUDE_ROOT}/util/ErrorChecking.cuh
    ${INCLUDE_ROOT}/util/NVTX.h
    ${INCLUDE_ROOT}/SAXPY.cuh
)
# Set source files (.c, .cpp, .cu)
SET(SRC_FILES
    util/ComputeCapability.cu
    main.cu
)
# Combine into single list.
set(ALL_SRC
    ${INCLUDE_FILES}
    ${SRC_FILES}
)

# Set the archive output directory - just for this project to the parent level?
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)

# Add the executable
add_executable(${EXECUTABLE_NAME} ${ALL_SRC})

# Set target specific properties, such as include directores and compiler flags.

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${INCLUDE_ROOT})


# If NVTX is enabled, set include/link args.
if(USE_NVTX AND NVTX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_NVTX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NVTX")

    target_include_directories(${EXECUTABLE_NAME} SYSTEM PRIVATE ${NVTX_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} ${NVTX_LIBRARIES})
endif()
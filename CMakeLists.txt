# Set the minimum cmake version to that which supports cuda natively.
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/modules/ ${CMAKE_MODULE_PATH})


# Name the project and set languages
project(cuda-concurrency-detection CUDA CXX)


# Define options.

#################

# SMS
SET(SMS "${SMS}" CACHE STRING "Compute Capabilities to build" FORCE)
# set_property(SMS PROPERTY HELPSTRING "List of compute capabilities to target. I.e. 61 70")


# USE_NVTX
option(USE_NVTX
"Enable NVTX markers for improved profiling (if available). Implied by profile builds"
ON
)


#################





# Optional dependency Checks
if(USE_NVTX)
    # Find the nvtx library using custom cmake module
    find_package(NVTX)
    # If not found, disable.
    if(NOT NVTX_FOUND)
        message("NVTX Not found, Setting USE_NVTX=OFF")
        SET(USE_NVTX "OFF")    
    endif()
endif()


# Include the src sub project 
add_subdirectory(src)

if(BUILD_TESTS)
    message("@todo")
endif()

# set(PROJECTNAME "main")

# add_executable(${PROJECTNAME} ${ALL_SRC})

# target_include_directories(${PROJECTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

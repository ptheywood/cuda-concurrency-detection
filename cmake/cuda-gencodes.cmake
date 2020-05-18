# Build a list of gencode arguments, based on CUDA verison.
# Accepts user override via SMS

# Check if any have been provided by the users
string(LENGTH "${SMS}" SMS_LENGTH)

# Define the default compute capabilites incase not provided by the user
set(DEFAULT_SMS "20;35;50;60;70;80;")


# Get the valid options for the current compiler.
# Run nvcc --help to get the help string which contains all valid compute_ sm_ for that version.
execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
# Match all comptue_XX or sm_XXs
string(REGEX MATCHALL "'(sm|compute)_[0-9]+'" SUPPORTED_SMS "${NVCC_HELP_STR}" )
# Strip just the numeric component
string(REGEX REPLACE "'(sm|compute)_([0-9]+)'" "\\2" SUPPORTED_SMS "${SUPPORTED_SMS}" )
# Remove dupes and sort to build the correct list of supported sms.
list(REMOVE_DUPLICATES SUPPORTED_SMS)
list(REMOVE_ITEM SUPPORTED_SMS "")
list(SORT SUPPORTED_SMS)

# Update defaults to only be those supported
foreach(SM IN LISTS DEFAULT_SMS)
    if (NOT SM IN_LIST SUPPORTED_SMS)
        list(REMOVE_ITEM DEFAULT_SMS "${SM}")
    endif()
endforeach()


if(NOT SMS_LENGTH EQUAL 0)
    # Convert user provided to a list.
    string (REPLACE " " ";" SMS "${SMS}")
    string (REPLACE "," ";" SMS "${SMS}")

    list(LENGTH SMS SMS_COUNT)

    # Validate the list.
    foreach(SM IN LISTS SMS)
        if (NOT SM IN_LIST SUPPORTED_SMS)
            message(WARNING "Compute Capability ${SM} not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION} and is being ignored.\nChoose from: ${SUPPORTED_SMS}")
            list(REMOVE_ITEM SMS "${SM}")
        endif()
    endforeach()

    # @todo - validate that the sms provided are supported by the compiler
endif()

# If the list is empty post validation, set it to the (validated) defaults
list(LENGTH SMS SMS_LENGTH)
if(SMS_LENGTH EQUAL 0)
    set(SMS ${DEFAULT_SMS})
endif()

# Remove duplicates, empty items and sort in ascending order.
list(REMOVE_DUPLICATES SMS)
list(REMOVE_ITEM SMS "")
list(SORT SMS)

# If the list is somehow empty now, do not set any gencodes arguments, instead using the compiler defaults.
list(LENGTH SMS SMS_LENGTH2)
if(NOT SMS_LENGTH EQUAL 0)
    message(STATUS "Using Compute Capabilities: ${SMS}")
    SET(GENCODES_FLAGS)
    SET(MIN_CUDA_ARCH)
    # Convert to gencode arguments

    foreach(SM IN LISTS SMS)
        set(GENCODES_FLAGS "${GENCODES_FLAGS} -gencode arch=compute_${SM},code=sm_${SM}")
    endforeach()

    # Add the last arch again as compute_, compute_ to enable forward looking JIT
    list(GET SMS -1 LAST_SM)
    set(GENCODES_FLAGS "${GENCODES_FLAGS} -gencode arch=compute_${LAST_SM},code=compute_${LAST_SM}")

    # Get the minimum device architecture to pass through to nvcc to enable graceful failure prior to cuda execution.
    list(GET SMS 0 MIN_CUDA_ARCH)

    # Set the gencode flags on NVCC
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODES_FLAGS}")


    # Set the minimum arch flags for all compilers
    SET(CMAKE_CC_FLAGS "${CMAKE_C_FLAGS} -DMIN_COMPUTE_CAPABILITY=${MIN_CUDA_ARCH}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIN_COMPUTE_CAPABILITY=${MIN_CUDA_ARCH}")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMIN_COMPUTE_CAPABILITY=${MIN_CUDA_ARCH}")
else()
    message(STATUS "Using default CUDA ${CMAKE_CUDA_COMPILER_VERSION} Compute Capabilities")
endif()

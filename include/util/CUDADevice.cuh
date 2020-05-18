#ifndef INCLUDE_UTIL_CUDA_DEVICE_CUH_
#define INCLUDE_UTIL_CUDA_DEVICE_CUH_

#include <cstdio>
#include "cuda_runtime.h"

#include "util/ComputeCapability.cuh"
#include "util/ErrorChecking.cuh"
#include "util/NVTX.h"

namespace util {
class CUDADevice {

 public:

int index;

CUDADevice(int index) :
    index(index),
    fetchedProps(false),
    props({}) { }

~CUDADevice(){ }

/**
 * Use this device (if possible) returning status.
 */
bool use(){
    NVTX_RANGE("CUDADevice::use()");

    // Check the index is a non-negative integer, less than the number of devices available
    if(this->index < 0){
        fprintf(stderr, "Error: Device indices must be non-negative integers, not %d\n", this->index);
        return false;
    }

    int deviceCount = CUDADevice::cudaDeviceCount();
    if(this->index >= deviceCount){
        fprintf(stderr, "Error: Device %d is invalid, only %d devices available.\n", this->index, deviceCount);
        return false;
    }

    // Ensure the capability is ok for this executable.
    // @todo cc might as well become part of this class.
    util::ComputeCapability::checkComputeCapability(this->index);


    // Attempt to set the device
    cudaError_t status = cudaSetDevice(this->index);
    if(cudaSuccess != status){
        fprintf(stderr, "Error: Cuda errror while setting device %d: %s\n", this->index, cudaGetErrorString(status));
        return false;
    }
    
    // Otherwise get props and establish the context?
    cudaDeviceProp &p = this->getDeviceProperties();
    printf("Using device %d: %s CC%d.%d with %d multiprocessors\n", this->index, p.name, p.major, p.minor, p.multiProcessorCount);

    // Establish the context
    CUDA_CALL(cudaFree(nullptr));


    return true;
}

void reset(){
    // @todo - could check for allocated memory here to nullptr?
    CUDA_CALL(cudaDeviceReset());
}

cudaDeviceProp & getDeviceProperties(){
    if(!this->fetchedProps){
        CUDA_CALL(cudaGetDeviceProperties(&this->props, this->index));
        this->fetchedProps = true;
    }
    return this->props;
}



/**
 * Get the number of cuda devices in the system
 */
static int cudaDeviceCount(){
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

 private:
    bool fetchedProps;
    cudaDeviceProp props;

};  // class CUDADevice
}  // namespace util
#endif  // INCLUDE_UTIL_CUDA_DEVICE_CUH_

#include "util/ComputeCapability.cuh"

#include "util/ErrorChecking.cuh"

int util::ComputeCapability::getComputeCapability(int deviceIndex) {
    int major = 0;
    int minor = 0;

    // Bail out 
    if (deviceIndex < 0) {
        return util::ComputeCapability::INVALID_CC;
    }

    // Ensure deviceIndex is valid.
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceIndex >= deviceCount) {
        throw util::ComputeCapability::INVALID_CC;;
    }
    // Load device attributes
    CUDA_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
    CUDA_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
    // Compute the arch integer value.
    int arch = (10 * major) + minor;
    return arch;
}

int util::ComputeCapability::minimumCompiledComputeCapability() {
    #if defined(MIN_COMPUTE_CAPABILITY)
        return MIN_COMPUTE_CAPABILITY;
    #else
        // Return 0 as a default minimum?
        return 0;
    #endif
}

bool util::ComputeCapability::checkComputeCapability(int deviceIndex) {
    // If the compile time minimum architecture is defined, fetch the device's compute capability and check that the executable (probably) supports this device.
    #if defined(MIN_COMPUTE_CAPABILITY)
        if (getComputeCapability(deviceIndex) < MIN_COMPUTE_CAPABILITY) {
            return false;
        } else {
            return true;
        }
    #else
        // If not defined, we cannot make a decision so assume it will work?
        return true;
    #endif
}

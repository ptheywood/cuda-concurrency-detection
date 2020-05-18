#ifndef INCLUDE_UTIL_ERROR_CHECKING_CUH_
#define INCLUDE_UTIL_ERROR_CHECKING_CUH_

#include <cuda_runtime.h>
#include <cstdio>

namespace util {
namespace ErrorChecking {

#define CUDA_CALL(code) { util::ErrorChecking::cudaCall((code), __FILE__, __LINE__); }
inline void cudaCall(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAError @ %s::%d - %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK() { util::ErrorChecking::cudaCheck(__FILE__, __LINE__); }
inline void cudaCheck(const char *file, int line) {
#ifdef _DEBUG
    util::ErrorChecking::cudaCall(cudaDeviceSynchronize(), file, line);
#endif
    util::ErrorChecking::cudaCall(cudaPeekAtLastError(), file, line);
}


}  // namespace ErrorChecking
}  // namespace util

#endif  // INCLUDE_UTIL_ERROR_CHECKING_CUH_

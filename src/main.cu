#include <cstdio>

#include "util/CUDADevice.cuh"
#include "util/NVTX.h"
#include "SAXPY.cuh"

int main(int argc, char * argv[]){
    NVTX_RANGE("main");

    // @todo - make this configurable.
    const int DEVICE = 0;
    const int BATCHES = 4;
    const int REPEATS = 1;
    const int TOTAL_ELEMENTS = 2 << 20;
    // const int TOTAL_ELEMENTS = (2 << 8) + 1;
    const float A = 2.0f;

    util::CUDADevice device = util::CUDADevice(DEVICE);

    bool status = device.use();
    if(!status){
        printf("Error: Could not use device\n");
        return EXIT_FAILURE;
    }
    
    // Solve the problem fo the total number of elements, using a number of streams and attempt to detect concurrency?

    printf("SAXPY on %d elements, in %d batches, using device %d\n", TOTAL_ELEMENTS, BATCHES, DEVICE);
    SAXPY::SAXPY saxpy = SAXPY::SAXPY();
    saxpy.allocate(TOTAL_ELEMENTS);
    saxpy.launch(A, BATCHES, REPEATS);
    saxpy.check(1);
    saxpy.deallocate();


    device.reset();
}
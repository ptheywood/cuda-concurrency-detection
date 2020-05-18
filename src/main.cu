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
    const int SEQUENTIAL_BATCHES = 1;
    const int CONCURRENT_BATCHS = BATCHES;
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
    
    saxpy.allocate(TOTAL_ELEMENTS, BATCHES);
    saxpy.launch(A, 1, REPEATS, false);


    printf("---- No streams\n");
    for(int concurrent_batches = SEQUENTIAL_BATCHES; concurrent_batches <= CONCURRENT_BATCHS; concurrent_batches++){
        if(concurrent_batches == SEQUENTIAL_BATCHES){
            printf("---- Sequential\n");
        } else if (concurrent_batches == CONCURRENT_BATCHS) {
            printf("---- Max Conccurency\n");
        } else {
            printf("---- Some Concurrent %d at a time\n", concurrent_batches);
        }
        saxpy.launch(A, concurrent_batches, REPEATS, true);
        // saxpy.check(1);
        // printf("----\n");
    }
    // saxpy.launch(A, BATCHES, CONCURRENT_BATCHS, REPEATS);
    // saxpy.check(1);
    saxpy.deallocate();


    device.reset();
}
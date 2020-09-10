#include <cstdio>

#include "util/CUDADevice.cuh"
#include "util/NVTX.h"
#include "SAXPY.cuh"

int main(int argc, char * argv[]){
    NVTX_RANGE("main");

    // @todo - make this configurable.
    const int DEVICE = 0;
    const int BATCHES = 4;
    const float A = 2.0f;

    // Total elements should corerspond to the device in question.
    // For a 1070, kernel uses 1024 threads per block, 15Ms so 30720 is the maximum thread block which is guaranteed to achieve concurrency. 
    // Larger problems shoudl be detectable with concurrency so long as there is atleast some concurrency however.
    const int TOTAL_ELEMENTS = 1048576;
    // const int TOTAL_ELEMENTS = 65536;
    // const int TOTAL_ELEMENTS = 30720;
    // const int TOTAL_ELEMENTS = 16384;
    // const int TOTAL_ELEMENTS = 2048;


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
    
    // Time launching in the default stream.
    saxpy.launch(A, 1);
    // Get the non-streamed time.
    float reference_ms = saxpy.getElapsedMillis();
    printf("reference_ms: %f\n", reference_ms);

    // Concurrency is achieved if the batched_ms is less than the reference, with some room for for error (which needs to be relative to the reference runtime?)
    // Lets say 10%. 
    const float CONCURRENCY_EPSILON_PERCENT = 0.1;
    const float concurrency_threshold_time = (1-CONCURRENCY_EPSILON_PERCENT) * reference_ms;
    printf("concurrency_threshold_ms: %f\n", concurrency_threshold_time);

    printf("---- No streams\n");
    for(int streams_to_use = 1; streams_to_use <= BATCHES; streams_to_use++){
        printf("Launching using %d/%d streams\n", streams_to_use, BATCHES);
        saxpy.launch(A, streams_to_use);
        // Get how long the batched time took.
        float batched_ms = saxpy.getElapsedMillis();


        bool concurrency_achieved = batched_ms < concurrency_threshold_time;
        printf("batched_ms: %f\n", batched_ms);
        printf("concurrency achieved? %d\n", concurrency_achieved);
    }

    // saxpy.check(1);
    saxpy.deallocate();


    device.reset();
}
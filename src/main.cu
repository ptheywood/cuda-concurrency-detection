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
    const int REPEATS = 5;

    // Total elements should corerspond to the device in question.
    // For a 1070, kernel uses 1024 threads per block, 15Ms so 30720 is the maximum thread block which is guaranteed to achieve concurrency. 
    // Larger problems shoudl be detectable with concurrency so long as there is atleast some concurrency however.
    // const int TOTAL_ELEMENTS = 1048576;
    const int TOTAL_ELEMENTS = 65536;
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
    

    // Get an average reference time.
    std::vector<float> reference_ms = std::vector<float>();
    float total_reference_ms = 0.f;
    for(int i = 0; i < REPEATS + 1; i++){
        // Launch with a single stream
        saxpy.launch(A, 1);
        // Don't time the first launch. 
        if(i > 0){
            // Get the elapsed time
            float ms = saxpy.getElapsedMillis();
            reference_ms.push_back(ms);
            total_reference_ms += ms;
        }
    }
    float mean_reference_ms = total_reference_ms / REPEATS;

    // Time launching in the default stream.
    // Get the non-streamed time.
    printf("mean_reference_ms: %f\n", mean_reference_ms);

    // 1% as a speedup threshold? Or should it be an amount of time?
    const float speedup_threshold = 1.01;

    for(int streams_to_use = 1; streams_to_use <= BATCHES; streams_to_use++){
        printf("Launching using %d/%d streams\n", streams_to_use, BATCHES);

        std::vector<float> batched_ms = std::vector<float>();
        float total_batched_ms = 0.f;
        for(int i = 0; i < REPEATS + 1; i++){
            // Launch with one or more streams
            saxpy.launch(A, streams_to_use);
            // Don't time the first launch. 
            if(i > 0){
                // Get the elapsed time
                float ms = saxpy.getElapsedMillis();
                batched_ms.push_back(ms);
                total_batched_ms += ms;
            }
        }
        float mean_batched_ms = total_batched_ms / REPEATS;


        float speedup = mean_reference_ms / mean_batched_ms;


        bool concurrency_achieved = speedup > speedup_threshold;
        printf("mean_batched_ms: %f\n", mean_batched_ms);
        printf("speedup %f, concurrency achieved? %d\n", speedup, concurrency_achieved);
    }

    // saxpy.check(1);
    saxpy.deallocate();


    device.reset();
}
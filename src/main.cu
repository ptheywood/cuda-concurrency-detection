#include <cstdio>

#include "util/CUDADevice.cuh"
#include "util/NVTX.h"
#include "SAXPY.cuh"

/*
    General notes on detecting concurrency:

    1. Individual kernels need to take enough time to be accurately measured. I.e. 10/100ms.
    2. Total problem size (across all streams) should be sufficiently small for the device being used (for higher speedup from concurrency).
    3. The more streams being used, the more potential concurrency
    4. Time each group of work multiple times and take an average
        + Consider dismissing the first timed kernel as this seems to be slower on average (power state?)
    5. Time against a single non-default stream as a reference, + use a threshold to ensure speedup is non-trivial.
        + Very larger problems benefit from streams even without significant overlap (imbalance?)
    6. Only use timing based on default stream events. Stream-based events are unreliable (see nv docs)
        + I tried using stream based timing, but it did not help.
    7. If timing differnt kernels with different runtimes, the longest running will take the same duration. Try to balance work between concurrent kernels.
*/


/*
 * multi grids per sm testing.
 * 
 * Wil an SM run blocks from separate grids (streams) at the same time? (if resources allow it)
 *
 * This should be testable with 2 kernel launches (more might make measuring speeup easier and still work, but might be more diffiuclt)
 * 
 * Launch a kernel which will places 1 block per SM on the device.
 *     Do not use the full SM - 1 or 2 warps per block, 1 block per grid
 *     Use more than 50% of the SM's shared memory - this will/should force the work to be spread over all SMs in the device, even when not launching a full device worth of work.
 * Launch a second kernel  which will then slot into the SMs
 *     Needs to fit in what is left of the SM in terms of compute - so 1 or 2 warps will definately fit
 *     Might need to use more than half the remaining shared memory, so that they get spread over all SMs.
 *          This might not be strictly neccesary, as even if they do not spread over all SMs, if *any* of them run at the same time as the other kernel we should see a speedup even if they all go in the first SM.
*/ 

// Maximum number of resident grids per device.
// 32: 35, 37. 50., 52 61
// 16: 53 62 72
// 128: 60 70 75 80 86

// cudaDeviceProp::maxBlocksPerMultiProcessor


int main(int argc, char * argv[]){
    NVTX_RANGE("main");

    // @todo - make this configurable.
    const int DEVICE = 0;
    const float A = 2.0f;
    const int REPEATS = 1;

    // Total elements should corerspond to the device in question.
    // For a 1070, kernel uses 1024 threads per block, 15Ms so 30720 is the maximum thread block which is guaranteed to achieve concurrency. 
    // Larger problems shoudl be detectable with concurrency so long as there is atleast some concurrency however.
    // const int TOTAL_ELEMENTS = 1048576;
    // const int TOTAL_ELEMENTS = 65536;
    // const int TOTAL_ELEMENTS = 30720;
    // const int TOTAL_ELEMENTS = 16384;
    // const int TOTAL_ELEMENTS = 8192;
    // const int TOTAL_ELEMENTS = 2048;


    util::CUDADevice device = util::CUDADevice(DEVICE);

    auto props = device.getDeviceProperties();
    int multiProcessorCount = props.multiProcessorCount;
    int maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    int warpSize = props.warpSize;
    int maxWarpsPerMultiProcessor = maxThreadsPerMultiProcessor / warpSize;
    int maxBlocksPerMultiProcessor = props.maxBlocksPerMultiProcessor;
    size_t sharedMemPerBlock = props.sharedMemPerBlock;
    size_t sharedMemPerMultiprocessor = props.sharedMemPerMultiprocessor;
    int max_resident_threads = multiProcessorCount * maxThreadsPerMultiProcessor;

    printf("GPU: %s SM_%d%d\n", props.name, props.major, props.minor);
    printf("multiProcessorCount %d\n", multiProcessorCount);
    printf("maxThreadsPerMultiProcessor %d\n", maxThreadsPerMultiProcessor);
    printf("maxWarpsPerMultiProcessor %d\n", maxWarpsPerMultiProcessor);
    printf("maxBlocksPerMultiProcessor %d\n", maxBlocksPerMultiProcessor);
    printf("sharedMemPerBlock %zu\n", sharedMemPerBlock);
    printf("sharedMemPerMultiprocessor %zu\n", sharedMemPerMultiprocessor);
    printf("max_resident_threads %d\n", max_resident_threads);
    printf("\n");
    
    // Calculate a problem size that will use all of the sahred memory using as few warps as possible.
    size_t shared_memory_per_block_full_shared = sharedMemPerBlock;
    int blocks_per_sm_full_shared = sharedMemPerMultiprocessor / shared_memory_per_block_full_shared;
    int gridssize_full_shared_full_device = multiProcessorCount * blocks_per_sm_full_shared;
    int blocksize_full_shared = warpSize;
    int total_threads_full_shared = (blocksize_full_shared * blocks_per_sm_full_shared) * gridssize_full_shared_full_device;
    printf("blocks_per_sm_full_shared %d\n", blocks_per_sm_full_shared);
    printf("gridssize_full_shared_full_device %d\n", gridssize_full_shared_full_device);
    printf("blocksize_full_shared %d\n", blocksize_full_shared);
    printf("total_threads_full_shared %d\n", total_threads_full_shared);
    printf("\n");

    
    // Calculate a problemsize that will lead to 1 block per 
    int blocks_per_sm_no_shared = 1;
    int gridssize_no_shared_full_device = multiProcessorCount * blocks_per_sm_no_shared;
    int blocksize_no_shared = warpSize;
    int total_threads_no_shared = (blocksize_no_shared * blocks_per_sm_no_shared) * gridssize_no_shared_full_device;
    printf("blocks_per_sm_no_shared %d\n", blocks_per_sm_no_shared);
    printf("gridssize_no_shared_full_device %d\n", gridssize_no_shared_full_device);
    printf("blocksize_no_shared %d\n", blocksize_no_shared);
    printf("total_threads_no_shared %d\n", total_threads_no_shared);
    printf("\n");


    // Error if the device can't do enough blocks for this.
    if(maxBlocksPerMultiProcessor < blocks_per_sm_full_shared + blocks_per_sm_no_shared){
        printf("Error: Device doesn's support enough blocks per multiprocessor\n");
        exit(EXIT_FAILURE);
    }

    // Error if the device can't support this many threads in flight.
    if(max_resident_threads < total_threads_full_shared + total_threads_no_shared){
        printf("Error: Device does not support enough threads for this test %d < %d\n", max_resident_threads, total_threads_full_shared + total_threads_no_shared);
        exit(EXIT_FAILURE);
    }


    float percent_sm_full_shared = (blocks_per_sm_full_shared * sharedMemPerBlock ) / (float) sharedMemPerMultiprocessor;
    printf("Grids:\n");
    printf("     shared: (%d,%d) %d. %d blocks/sm, %.2f%% shared\n", gridssize_full_shared_full_device, blocksize_full_shared, total_threads_full_shared, blocks_per_sm_full_shared, percent_sm_full_shared);
    printf("  no shared: (%d,%d) %d. %d blocks/sm, %.2f%% shared\n", gridssize_no_shared_full_device, blocksize_no_shared, total_threads_no_shared, blocks_per_sm_no_shared, 0.f);
    printf("\n");

    // Calculate thte total number of elements we are using etc for allocations.
    int TOTAL_ELEMENTS = total_threads_full_shared + total_threads_no_shared;
    const int BATCHES = 2;

    std::vector<int> gridSizes = {gridssize_full_shared_full_device, gridssize_no_shared_full_device};
    std::vector<int> blockSizes = {blocksize_full_shared, blocksize_no_shared};
    std::vector<size_t> sharedMemoryPerBlockSizes = {shared_memory_per_block_full_shared, 0u};



    printf("TOTAL_ELEMENTS %d\n", TOTAL_ELEMENTS);
    printf("BATCHES %d\n", BATCHES);

    printf("\n");
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
        saxpy.launch(A, 1, gridSizes, blockSizes, sharedMemoryPerBlockSizes);
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
    int streams_to_use = BATCHES;
    
    printf("Launching using %d/%d streams\n", streams_to_use, BATCHES);

    std::vector<float> batched_ms = std::vector<float>();
    float total_batched_ms = 0.f;
    for(int i = 0; i < REPEATS + 1; i++){
        // Launch with one or more streams
        saxpy.launch(A, streams_to_use, gridSizes, blockSizes, sharedMemoryPerBlockSizes);
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

    // saxpy.check(1);
    saxpy.deallocate();


    device.reset();
}
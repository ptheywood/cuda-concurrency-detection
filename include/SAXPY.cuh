#ifndef INCLUDE_SAXPY_CUH
#define INCLUDE_SAXPY_CUH

#include <cstring>
#include <memory>
#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace SAXPY {

// Make the kernel take longer with pointless repetition
static constexpr const int INNER_REPEATS = 128;

__global__ void saxpy_kernel(int n, float a, float *x, float *y, float *z) {
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
        for(int r = 0; r < INNER_REPEATS; r++){
            z[i] = a * x[i] + y[i];
        }
    }
}


class SAXPY {

public:
SAXPY() : 
    deviceIndex(0),
    batches(0),
    n(0),
    a(0),
    h_x(nullptr),
    h_y(nullptr),
    h_z(nullptr),
    d_x(nullptr),
    d_y(nullptr),
    d_z(nullptr),
    allocatedLength(0),
    g_start(),
    g_stop(),
    startEvents(std::vector<cudaEvent_t>()),
    stopEvents(std::vector<cudaEvent_t>()),
    streams(std::vector<cudaStream_t>()) { }
~SAXPY() { }

bool allocate(const int n, const int batches) {
    NVTX_RANGE("saxpy::allocate");
    // @todo use the device? 

    this->n = n;
    this->batches = batches;
    if(n <= 0){
        fprintf(stderr, "Error: n must be >= 0. n=%d\n", n);
        return false;
    }
    size_t bytes = this->n * sizeof(float);

    this->h_x = (float*)std::malloc(bytes);
    this->h_y = (float*)std::malloc(bytes);
    this->h_z = (float*)std::malloc(bytes);

    CUDA_CALL(cudaMalloc((void**)&this->d_x, bytes));
    CUDA_CALL(cudaMalloc((void**)&this->d_y, bytes));
    CUDA_CALL(cudaMalloc((void**)&this->d_z, bytes));

    if(this->h_x == nullptr || this->h_y == nullptr || this->h_z == nullptr || this->d_x == nullptr || this->d_y == nullptr || this->d_z == nullptr){

        printf("Error: allocation failure %p %p %p %p %p %p\n", this->h_x, this->h_y, this->h_z, this->d_x, this->d_y, this->d_z);
        this->deallocate();
        return false;
    } else {
        allocatedLength = this->n;
        // Memset.
        // std::memset(this->h_x, 0, bytes);
        // std::memset(this->h_y, 0, bytes);
        std::fill(this->h_x, this->h_x + this->n, 1.0f);
        std::fill(this->h_y, this->h_y+ this->n, 2.0f);
        std::memset(this->h_z, 0, bytes);
        CUDA_CALL(cudaMemset(this->d_x, 0, bytes));
        CUDA_CALL(cudaMemset(this->d_y, 0, bytes));
        CUDA_CALL(cudaMemset(this->d_z, 0, bytes));


        // Create streams
        NVTX_PUSH("createStreams");
        for(int b = 0; b < batches; b++){
            cudaStream_t stream;
            CUDA_CALL(cudaStreamCreate(&stream));
            streams.push_back(stream);
        }
        NVTX_POP();

        // Create events
        NVTX_PUSH("createEvents");
        
        
        CUDA_CALL(cudaEventCreate(&g_start));
        CUDA_CALL(cudaEventCreate(&g_stop));
        for(int b = 0; b < batches; b++){
            cudaEvent_t event;
            CUDA_CALL(cudaEventCreate(&event));
            startEvents.push_back(event);
        }
        for(int b = 0; b < batches; b++){
            cudaEvent_t event;
            CUDA_CALL(cudaEventCreate(&event));
            stopEvents.push_back(event);
        }
        NVTX_POP();


        return true;
    }
    
}
void deallocate() {
    NVTX_RANGE("saxpy::deallocate");
    if(h_x != nullptr){
        std::free(h_x);
        h_x = nullptr;
    }
    if(h_y != nullptr){
        std::free(h_y);
        h_y = nullptr;
    }
    if(d_x != nullptr){
        CUDA_CALL(cudaFree(d_x));
        d_x = nullptr;
    }
    if(d_y != nullptr){
        CUDA_CALL(cudaFree(d_y));
        d_y = nullptr;
    }

    // Destroy events
    NVTX_PUSH("eventDestroy");
    for (auto event : stopEvents) {
        CUDA_CALL(cudaEventDestroy(event));
    }
    stopEvents.clear();
    for (auto event : startEvents) {
        CUDA_CALL(cudaEventDestroy(event));
    }
    startEvents.clear();
    CUDA_CALL(cudaEventDestroy(g_stop));
    CUDA_CALL(cudaEventDestroy(g_start));
    NVTX_POP();

    // Destroy streams
    NVTX_PUSH("streamDestroy");
    for (auto stream : streams) {
        CUDA_CALL(cudaStreamDestroy(stream));
    }
    streams.clear();
    NVTX_POP();
}

void launch(const float a, const int concurrent_batches, const int repeats, bool use_streams) {
    NVTX_RANGE("saxpy::launch");
    printf("saxpy::launch %d/%d\n", concurrent_batches, batches);
    this->a = a;
    if(this->allocatedLength > 0){
        size_t size = this->n * sizeof(float);

        // Copy data in
        NVTX_PUSH("h2s");
        CUDA_CALL(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
        NVTX_POP();
        
        // Compute batch sizes and therefore kernel args / sizes.
        int batchSize = ceil(this->n / (float)batches);
        // Batch size must be a mult of 32? The final batch size will (might) be smaller.
        int rem32 = batchSize % 32;
        if (rem32 != 0) {
            batchSize = batchSize + 32 - rem32;
        }

        std::vector<float> eventMillis = std::vector<float>();
        for(int b = 0; b < batches; b++){
            eventMillis.push_back(0.0);
        }
        
        printf("n %d, batches %d, batchSize %d, tot %d\n", this->n, batches, batchSize, batches * batchSize);
        
        int minGridSize = 0;
        int blockSize = 0;
        int gridSize = 0;
        
        
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ::SAXPY::saxpy_kernel));

        // Threads per block needs to be limited to only use a certain number of SMS. 
        // minGridSize is one way of knowing this. I.e. for 1024 threads per block, a 1070 requies 30 grids (15 sms, 2048 threads per sm.)
        
        int maxGridSizePerBatch = minGridSize / concurrent_batches;
        gridSize = (batchSize + blockSize - 1) / blockSize;
        gridSize = std::min(gridSize, maxGridSizePerBatch);
        
        // @todo grid stride? - only launch as many blocks as needed to fill part of the device - i.e. clamp.
        // @topo launching far too many threads at small block sizes (i.e. 1024 threads per block, even when doing multiple blocks.)

        printf("n=%d. Launching %d batches of %d as blockSize %d gridSize %d\n", this->n, batches, batchSize, blockSize, gridSize);
        
        for(int r = 0; r < repeats; r++){
            // Record an event in the default stream before issuing any kernels.
            CUDA_CALL(cudaEventRecord(g_start));
            for(int b = 0; b < batches; b++){
                // int s = b % concurrent_batches;
                // printf("b %d, batches %d, concurrent_batches %d, s %d\n", b, batches, concurrent_batches, s);
                cudaStream_t stream = streams.at(b);
                if (!use_streams){
                    stream = 0;
                }
                cudaEvent_t start = startEvents.at(b);
                cudaEvent_t stop = stopEvents.at(b);
                int offset = b * batchSize;
                // @todo launching too many threads for the last batch?

                NVTX_PUSH("saxpy_kernel\n");
                // Record an event before per stream
                CUDA_CALL(cudaEventRecord(start, stream));
                ::SAXPY::saxpy_kernel<<<gridSize, blockSize, 0, stream>>>(
                    batchSize,
                    a,
                    this->d_x + offset,
                    this->d_y + offset,
                    this->d_z + offset
                );
                // Record an event after per stream
                CUDA_CALL(cudaEventRecord(stop, stream));
                NVTX_POP();

                // Add an arbitary device sync to break stuff.
                if (b % 2 == 0){
                    CUDA_CALL(cudaDeviceSynchronize());
                }
            }
            // Record an event in the default stream afterwards - this will block though...?
            CUDA_CALL(cudaEventRecord(g_stop));

            // Sync each stop evet
            for(int b = 0; b < batches; b++){
                cudaEvent_t stop = stopEvents.at(b);
                CUDA_CALL(cudaEventSynchronize(stop));
            }
            CUDA_CALL(cudaEventSynchronize(g_stop));

            float g2g_ms = 0.f;
            CUDA_CALL(cudaEventElapsedTime(&g2g_ms, g_start, g_stop));
            // Record the elapsed times
            float s2s_ms_sum = 0.f;
            float g2s_ms_sum = 0.f;
            float s2g_ms_sum = 0.f;
            for(int b = 0; b < batches; b++){
                auto start = startEvents.at(b);
                auto stop = stopEvents.at(b);
                float ms = eventMillis.at(b);
                CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
                float g2s_ms = 0.f;
                CUDA_CALL(cudaEventElapsedTime(&g2s_ms, g_start, stop));
                float s2g_ms = 0.f;
                CUDA_CALL(cudaEventElapsedTime(&s2g_ms, start, g_stop));
                
                s2s_ms_sum += ms;
                g2s_ms_sum += g2s_ms;
                s2g_ms_sum += s2g_ms;
                
                printf("b %d s2s %f, g2g %f g2s %f s2g %f\n", b, ms, g2g_ms, g2s_ms, s2g_ms);
            }
            printf("g2g   est. total %f mean %f\n", g2g_ms*batches, g2g_ms);
            printf("s2s events total %f mean %f\n", s2s_ms_sum, s2s_ms_sum / batches);
            printf("g2s events total %f mean %f\n", g2s_ms_sum, g2s_ms_sum / batches);
            printf("s2g events total %f mean %f\n", s2g_ms_sum, s2g_ms_sum / batches);
        }
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CHECK();

        // Copy out
        NVTX_PUSH("d2h");
        CUDA_CALL(cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost));
        NVTX_POP();
    }
}

void check(const int elementsToCheck){
    if(this->allocatedLength > 0){
        int e = std::min(elementsToCheck, this->allocatedLength);
        for(int i = 0; i < e; i++){
            printf("%d: %f + %f * %f = %f\n", i, this->a, this->h_x[i], this->h_y[i], this->h_z[i]);
        }
        if(e < this->n - 1){
            int i = this->n - 1;
            printf("...\n");
            printf("%d: %f + %f * %f = %f\n", i, this->a, this->h_x[i], this->h_y[i], this->h_z[i]);
        }
    }
}


private:
    int deviceIndex;
    int batches;
    int n;
    float a;
    float * h_x;
    float * h_y;
    float * h_z;
    float * d_x;
    float * d_y;
    float * d_z;
    int allocatedLength;
    cudaEvent_t g_start;
    cudaEvent_t g_stop;
    std::vector<cudaEvent_t> startEvents;
    std::vector<cudaEvent_t> stopEvents;
    std::vector<cudaStream_t> streams;
};

}  // namespace SAXPY


#endif  // INCLUDE_SAXPY_CUH

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
// static constexpr const int INNER_REPEATS = 32768;
static constexpr const int INNER_REPEATS = 1048576;


// Do not use const restrict, it makes it too fast.
// __global__ void saxpy_kernel(const int n, const float a, const float * __restrict__ x, const float * __restrict__ y, float * z) {
__global__ void saxpy_kernel(const int n, const float a, const float * x, const float * y, float * z) {
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
        for(int r = 0; r < INNER_REPEATS; r++){
            z[i] = a * x[i] + y[i];
        }
    }
}

__global__ void saxpy_kernel_smem(const int n, const float a, const float * x, const float * y, float * z) {
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
    start({}),
    stop({}),
    elapsedMillis(0.f),
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
        
        printf("creating events\n");
        CUDA_CALL(cudaEventCreate(&start));
        CUDA_CALL(cudaEventCreate(&stop));
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
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaEventDestroy(start));
    NVTX_POP();

    // Destroy streams
    NVTX_PUSH("streamDestroy");
    for (auto stream : streams) {
        CUDA_CALL(cudaStreamDestroy(stream));
    }
    streams.clear();
    NVTX_POP();
}
void launch(const float a, const int streamCount, std::vector<int> gridSizes, std::vector<int> blockSizes, std::vector<size_t> sharedMemoryPerBlockSizes) {
    NVTX_RANGE("saxpy::launch");
    printf("saxpy::launch %d %d %zu %zu %zu\n", streamCount, batches, gridSizes.size(), blockSizes.size(), sharedMemoryPerBlockSizes.size());

    this->a = a;
    if(this->allocatedLength > 0){
        size_t size = this->n * sizeof(float);

        // Copy data in
        NVTX_PUSH("h2s");
        CUDA_CALL(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
        NVTX_POP();
        
        std::vector<float> eventMillis = std::vector<float>();
        for(int b = 0; b < batches; b++){
            eventMillis.push_back(0.0);
        }
    

        
        // Record an event in the default stream before issuing any kernels.
        // Event timing is only reliable in the default stream (i.e. blocking), so time the whole process rather than per stream only. I.e. all layers.
        CUDA_CALL(cudaEventRecord(start));

        int offset = 0;

        // Iterate each batch, launching the kernel with the appropriate args.
        for(int b = 0; b < batches; b++){
            int blockSize = blockSizes.at(b);
            int gridSize = gridSizes.at(b);
            size_t sharedBytes = sharedMemoryPerBlockSizes.at(b);
            int batchSize = blockSize * gridSize;
            
            printf("streamCount %d\n", streamCount);
            
            // Use the default stream by default.
            cudaStream_t stream = 0;
            int streamIdx = 0;
            // If the number of streams to use is atleast 1, fetch the appropraite stream object.
            if(streamCount > 0) {
                streamIdx = b % streamCount;
                stream = streams.at(streamIdx);
            }
            printf("n=%d. batch %d/%d threads %d blockSize %d gridSize %d smem %zu stream %d %zu\n", this->n, b, batches, batchSize, blockSize, gridSize, sharedBytes, streamIdx, (size_t)stream);

            // @todo launching too many threads for the last batch?
            NVTX_PUSH("saxpy_kernel\n");

            if(sharedBytes > 0) {
                ::SAXPY::saxpy_kernel_smem<<<gridSize, blockSize, sharedBytes, stream>>>(
                    batchSize,
                    a,
                    this->d_x + offset,
                    this->d_y + offset,
                    this->d_z + offset
                );
            } else {
                ::SAXPY::saxpy_kernel<<<gridSize, blockSize, 0, stream>>>(
                    batchSize,
                    a,
                    this->d_x + offset,
                    this->d_y + offset,
                    this->d_z + offset
                );
            }
            CUDA_CHECK();
            // Record an event after per stream
            NVTX_POP();

            // Add an arbitary device sync to break stuff.
            // if (b % 2 == 0){
            //     CUDA_CALL(cudaDeviceSynchronize());
            // }

            // Update the offset for the next batch
            offset += batchSize;
        }
        // Record that all work has been issued by this point.
        CUDA_CALL(cudaEventRecord(stop));

        // Sync the stop event, (i.e. a device sync as in default stream
        CUDA_CALL(cudaEventSynchronize(stop));
        // Calculate how long between start and stop events in ms.
        CUDA_CALL(cudaEventElapsedTime(&elapsedMillis, start, stop));

        // CUDA_CALL(cudaDeviceSynchronize());
        // CUDA_CHECK();

        // printf("elapsed ms: %f\n", elapsedMillis);

        // Copy out for checking data.
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


float getElapsedMillis(){
    return this->elapsedMillis;
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
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedMillis;
    std::vector<cudaStream_t> streams;
};

}  // namespace SAXPY


#endif  // INCLUDE_SAXPY_CUH

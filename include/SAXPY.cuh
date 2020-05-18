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
    streamIndex(0),
    n(0),
    a(0),
    h_x(nullptr),
    h_y(nullptr),
    h_z(nullptr),
    d_x(nullptr),
    d_y(nullptr),
    d_z(nullptr),
    allocatedLength(0) { }
~SAXPY() { }
bool allocate(const int n) {
    NVTX_RANGE("saxpy::allocate");
    // @todo use the device? 

    this->n = n;
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
}

void launch(const float a, const int batches, const int repeats) {
    NVTX_RANGE("saxpy::launch");
    printf("saxpy::launch(%f, %d)\n", a, batches);
    this->a = a;
    if(this->allocatedLength > 0){
        size_t size = this->n * sizeof(float);

        // Create streams
        NVTX_PUSH("createStreams");
        std::vector<cudaStream_t> streams = std::vector<cudaStream_t>();
        for(int b = 0; b < batches; b++){
            cudaStream_t stream;
            CUDA_CALL(cudaStreamCreate(&stream));
            streams.push_back(stream);
        }
        NVTX_POP();

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

        
        
        printf("n %d, batches %d, batchSize %d, tot %d\n", this->n, batches, batchSize, batches * batchSize);
        
        int minGridSize = 0;
        int blockSize = 0;
        int gridSize = 0;
        
        
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ::SAXPY::saxpy_kernel));

        // Threads per block needs to be limited to only use a certain number of SMS. 
        // minGridSize is one way of knowing this. I.e. for 1024 threads per block, a 1070 requies 30 grids (15 sms, 2048 threads per sm.)
        
        int maxGridSizePerBatch = minGridSize / batches;
        gridSize = (batchSize + blockSize - 1) / blockSize;
        gridSize = std::min(gridSize, maxGridSizePerBatch);
        
        // @todo grid stride? - only launch as many blocks as needed to fill part of the device - i.e. clamp.
        // @topo launching far too many threads at small block sizes (i.e. 1024 threads per block, even when doing multiple blocks.)

        printf("n=%d. Launching %d batches of %d as blockSize %d gridSize %d\n", this->n, batches, batchSize, blockSize, gridSize);
        
        for(int r = 0; r < repeats; r++){
            for(int b = 0; b < batches; b++){
                cudaStream_t stream = streams.at(b);
                int offset = b * batchSize;
                // @todo launching too many threads for the last batch?

                NVTX_PUSH("saxpy_kernel\n");
                ::SAXPY::saxpy_kernel<<<gridSize, blockSize, 0, stream>>>(
                    batchSize,
                    a,
                    this->d_x + offset,
                    this->d_y + offset,
                    this->d_z + offset
                );
                NVTX_POP();
            }
        }
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CHECK();

        // Copy out
        NVTX_PUSH("d2h");
        CUDA_CALL(cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost));
        NVTX_POP();

        // Destroy streams
        NVTX_PUSH("streamDestroy");
        int b = 0;
        for (auto stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
            b++;
        }
        streams.clear();
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
    int streamIndex;
    int n;
    float a;
    float * h_x;
    float * h_y;
    float * h_z;
    float * d_x;
    float * d_y;
    float * d_z;
    int allocatedLength;
};

}  // namespace SAXPY


#endif  // INCLUDE_SAXPY_CUH

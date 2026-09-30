#include <cuda_runtime.h>


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <helper_cuda.h>
#include <helper_functions.h>

#define VERSION_MAJOR (CUDART_VERSION / 1000)
#define VERSION_MINOR (CUDART_VERSION % 100) / 10

const char *sSDKsample = "threadFenceReduction";

#if CUDART_VERSION >= 2020


#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;


template <unsigned int blockSize>
__device__ void reduceBlock(volatile float *sdata, float mySum, const unsigned int tid, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid]                       = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    const int vid = tid & (VEC - 1);

    float beta = mySum;
    float temp;

    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (vid < i) {
            temp = sdata[tid + i];
            beta += temp;
            sdata[tid] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocks(const float *g_idata, float *g_odata, unsigned int n, cg::thread_block cta)
{
    extern __shared__ float sdata[];

    
    
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float        mySum    = 0;

    
    
    
    while (i < n) {
        mySum += g_idata[i];

        
        
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i + blockSize];

        i += gridSize;
    }

    
    reduceBlock<blockSize>(sdata, mySum, tid, cta);

    
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    
    cg::thread_block cta = cg::this_thread_block();
    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
}


__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}


template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n)
{
    
    cg::thread_block cta = cg::this_thread_block();
    
    
    

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);

    
    
    

    if (gridDim.x > 1) {
        const unsigned int      tid = threadIdx.x;
        __shared__ bool         amLast;
        extern float __shared__ smem[];

        
        
        __threadfence();

        
        if (tid == 0) {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            
            
            amLast = (ticket == gridDim.x - 1);
        }

        cg::sync(cta);

        
        if (amLast) {
            int   i     = tid;
            float mySum = 0;

            while (i < gridDim.x) {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid, cta);

            if (tid == 0) {
                g_odata[0] = smem[0];

                
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }


extern "C" void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int  smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    
    if (isPow2(size)) {
        switch (threads) {
        case 512:
            reduceMultiPass<512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduceMultiPass<256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduceMultiPass<128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduceMultiPass<64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduceMultiPass<32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduceMultiPass<16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduceMultiPass<8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduceMultiPass<4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduceMultiPass<2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduceMultiPass<1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    }
    else {
        switch (threads) {
        case 512:
            reduceMultiPass<512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduceMultiPass<256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduceMultiPass<128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduceMultiPass<64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduceMultiPass<32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduceMultiPass<16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduceMultiPass<8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduceMultiPass<4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduceMultiPass<2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduceMultiPass<1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    }
}

extern "C" void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int  smemSize = threads * sizeof(float);

    
    if (isPow2(size)) {
        switch (threads) {
        case 512:
            reduceSinglePass<512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduceSinglePass<256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduceSinglePass<128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduceSinglePass<64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduceSinglePass<32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduceSinglePass<16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduceSinglePass<8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduceSinglePass<4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduceSinglePass<2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduceSinglePass<1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    }
    else {
        switch (threads) {
        case 512:
            reduceSinglePass<512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduceSinglePass<256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduceSinglePass<128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduceSinglePass<64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduceSinglePass<32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduceSinglePass<16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduceSinglePass<8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduceSinglePass<4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduceSinglePass<2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduceSinglePass<1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    }
}

#endif 

#else
#pragma comment(user, "CUDA 2.2 is required to build for threadFenceReduction")
#endif


bool runTest(int argc, char **argv);

extern "C"
{
    void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata);
    void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata);
}

#if CUDART_VERSION < 2020
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    printf("reduce(), compiler not supported, aborting tests\n");
}

void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    printf("reduceSinglePass(), compiler not supported, aborting tests\n");
}
#endif


int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    printf("%s Starting...\n\n", sSDKsample);

    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("GPU Device supports SM %d.%d compute capability\n\n", deviceProp.major, deviceProp.minor);

    bool bTestResult = false;

#if CUDART_VERSION >= 2020
    bTestResult = runTest(argc, argv);
#else
    print_NVCC_min_spec(sSDKsample, "2.2", "Version 185");
    exit(EXIT_SUCCESS);
#endif

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}


template <class T> T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c   = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1) {
        threads = 1;
        blocks  = 1;
    }
    else {
        threads = (n < maxThreads * 2) ? nextPow2(n / 2) : maxThreads;
        blocks  = max(1, n / (threads * 2));
    }

    blocks = min(maxBlocks, blocks);
}

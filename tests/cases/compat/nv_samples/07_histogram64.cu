#include <cuda_runtime.h>


#include <assert.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>


#ifndef HISTOGRAM_COMMON_H
#define HISTOGRAM_COMMON_H


#define HISTOGRAM64_BIN_COUNT  64
#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS              32
typedef unsigned int  uint;
typedef unsigned char uchar;


#define LOG2_WARP_SIZE 5U
#define WARP_SIZE      (1U << LOG2_WARP_SIZE)


#define SHARED_MEMORY_BANKS 16


#define HISTOGRAM64_THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)


#define WARP_COUNT 6


#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)


#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define UMUL(a, b)    ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))


extern "C" void histogram64CPU(uint *h_Histogram, void *h_Data, uint byteCount);

extern "C" void histogram256CPU(uint *h_Histogram, void *h_Data, uint byteCount);


extern "C" void initHistogram64(void);
extern "C" void initHistogram256(void);
extern "C" void closeHistogram64(void);
extern "C" void closeHistogram256(void);

extern "C" void histogram64(uint *d_Histogram, void *d_Data, uint byteCount);

extern "C" void histogram256(uint *d_Histogram, void *d_Data, uint byteCount);

#endif


typedef uint4 data_t;


#define SHARED_MEMORY_BANKS 16


inline __device__ void addByte(uchar *s_ThreadBase, uint data)
{
    s_ThreadBase[UMUL(data, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}


inline __device__ void addWord(uchar *s_ThreadBase, uint data)
{
    
    addByte(s_ThreadBase, (data >> 2) & 0x3FU);
    addByte(s_ThreadBase, (data >> 10) & 0x3FU);
    addByte(s_ThreadBase, (data >> 18) & 0x3FU);
    addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

__global__ void histogram64Kernel(uint *d_PartialHistograms, data_t *d_Data, uint dataCount)
{
    
    cg::thread_block cta = cg::this_thread_block();
    
    
    
    
    
    
    const uint threadPos = ((threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0)
                         | ((threadIdx.x & (SHARED_MEMORY_BANKS - 1)) << 2)
                         | ((threadIdx.x & (SHARED_MEMORY_BANKS * 3)) >> 4);

    
    __shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
    uchar           *s_ThreadBase = s_Hist + threadPos;


#pragma unroll

    for (uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++) {
        ((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;
    }

    
    
    
    cg::sync(cta);

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)) {
        data_t data = d_Data[pos];
        addWord(s_ThreadBase, data.x);
        addWord(s_ThreadBase, data.y);
        addWord(s_ThreadBase, data.z);
        addWord(s_ThreadBase, data.w);
    }

    
    cg::sync(cta);

    if (threadIdx.x < HISTOGRAM64_BIN_COUNT) {
        uchar *s_HistBase = s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

        uint sum = 0;
        uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

#pragma unroll

        for (uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++) {
            sum += s_HistBase[pos + 0] + s_HistBase[pos + 1] + s_HistBase[pos + 2] + s_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = sum;
    }
}


#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram64Kernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  data[MERGE_THREADBLOCK_SIZE];

    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
    }

    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        cg::sync(cta);

        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        d_Histogram[blockIdx.x] = data[0];
    }
}


static const uint MAX_PARTIAL_HISTOGRAM64_COUNT = 32768;
static uint      *d_PartialHistograms;


extern "C" void initHistogram64(void)
{
    assert(HISTOGRAM64_THREADBLOCK_SIZE % (4 * SHARED_MEMORY_BANKS) == 0);
    checkCudaErrors(cudaMalloc((void **)&d_PartialHistograms,
                               MAX_PARTIAL_HISTOGRAM64_COUNT * HISTOGRAM64_BIN_COUNT * sizeof(uint)));
}


extern "C" void closeHistogram64(void) { checkCudaErrors(cudaFree(d_PartialHistograms)); }


inline uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


inline uint iSnapDown(uint a, uint b) { return a - a % b; }

extern "C" void histogram64(uint *d_Histogram, void *d_Data, uint byteCount)
{
    const uint histogramCount = iDivUp(byteCount, HISTOGRAM64_THREADBLOCK_SIZE * iSnapDown(255, sizeof(data_t)));

    assert(byteCount % sizeof(data_t) == 0);
    assert(histogramCount <= MAX_PARTIAL_HISTOGRAM64_COUNT);

    histogram64Kernel<<<histogramCount, HISTOGRAM64_THREADBLOCK_SIZE>>>(
        d_PartialHistograms, (data_t *)d_Data, byteCount / sizeof(data_t));
    getLastCudaError("histogram64Kernel() execution failed\n");

    mergeHistogram64Kernel<<<HISTOGRAM64_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
        d_Histogram, d_PartialHistograms, histogramCount);
    getLastCudaError("mergeHistogram64() execution failed\n");
}

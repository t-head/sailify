#include <cuda_runtime.h>


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <helper_cuda.h>      
#include <helper_functions.h> 


typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct
{
    unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct
{
    unsigned int l, a;
} LA32_misaligned;

typedef struct
{
    unsigned int r, g, b;
} RGB32_misaligned;

typedef struct
{
    unsigned int r, g, b, a;
} RGBA32_misaligned;


typedef struct __align__(4)
{
    unsigned char r, g, b, a;
} RGBA8;

typedef unsigned int I32;

typedef struct __align__(8)
{
    unsigned int l, a;
} LA32;

typedef struct __align__(16)
{
    unsigned int r, g, b;
} RGB32;

typedef struct __align__(16)
{
    unsigned int r, g, b, a;
} RGBA32;


typedef struct __align__(16)
{
    RGBA32 c1, c2;
} RGBA32_2;


int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


int iDivDown(int a, int b) { return a / b; }


int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }


int iAlignDown(int a, int b) { return a - a % b; }


template <class TData> __global__ void testKernel(TData *d_odata, TData *d_idata, int numElements)
{
    const int tid        = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    for (int pos = tid; pos < numElements; pos += numThreads) {
        d_odata[pos] = d_idata[pos];
    }
}


template <class TData> int testCPU(TData *h_odata, TData *h_idata, int numElements, int packedElementSize)
{
    for (int pos = 0; pos < numElements; pos++) {
        TData src = h_idata[pos];
        TData dst = h_odata[pos];

        for (int i = 0; i < packedElementSize; i++)
            if (((char *)&src)[i] != ((char *)&dst)[i]) {
                return 0;
            }
    }

    return 1;
}


const int MEM_SIZE       = 50000000;
const int NUM_ITERATIONS = 32;


unsigned char *d_idata, *d_odata;

unsigned char      *h_idataCPU, *h_odataGPU;

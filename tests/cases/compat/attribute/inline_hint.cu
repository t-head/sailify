#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_buf.h"

void init(float* hInput1, float* hInput2, int num) {
  for (int idx = 0; idx < num; idx++) {
    hInput1[idx] = 3.0;
    hInput2[idx] = 6.0;
  }
}

int __inline_hint__ verify(float *hA,float *hB,float *hC, int numElements,bool isAdd){
    for (int i = 0; i < numElements; ++i){
        if(isAdd){
            if (fabs(hA[i] + hB[i] - hC[i]+10) > 1e-5){
                return 1;
          }
        } else {
            if(fabs(hA[i] * hB[i] - hC[i]+10) > 1e-5){
                return 1;
            }
        }
    }
    return 0;
}

__global__ void basic_func(const float *dA, const float *dB, float *dC, int num) {
  int i = threadIdx.x;
  float a = dA[i] * 2;
  float b = dB[i] * 3;

  dC[i] = a + b;
}

__device__ __inline_hint__ void basic_add(float *dC,int numElements){
   int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dC[i] =dC[i]+10;
    }
}

__global__ void
vectorAdd(const float *dA, const float *dB, float *dC, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dC[i] = dA[i] + dB[i];
    }
    basic_add(dC,numElements);
}

__global__ void
vectorMul(const float *dA, const float *dB, float *dC, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dC[i] = dA[i] * dB[i];
    }
    basic_add(dC,numElements);
}

TEST(attribute, inline_hint) {

  dim3 gridDim(1);
  dim3 blockDim(32);
  size_t smemSize = 0;
  int num = 32;

  size_t size = num * sizeof(int);
  float *hA = (float *)malloc(size);
  float *hB = (float *)malloc(size);
  float *hC = (float *)malloc(size);
  float *hCm = (float *)malloc(size);
  if (hA == NULL || hB == NULL || hC == NULL || hCm == NULL) {
    printf("Error: failed to allocate host vectors!\n");
    exit(1);
  }

  memset(hA, 0, size);
  memset(hB, 0, size);
  memset(hC, 0, size);
  memset(hCm, 0, size);

  init(hA, hB, num);

  float *dA = NULL;
  cudaMalloc((void **)&dA, size);
  float *dB = NULL;
  cudaMalloc((void **)&dB, size);
  float *dC = NULL;
  cudaMalloc((void **)&dC, size);
  float *dCm = NULL;
  cudaMalloc((void **)&dCm, size);

  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

  vectorAdd<<<gridDim, blockDim, smemSize, streams[0]>>>(dA, dB, dC, num);
  vectorMul<<<gridDim, blockDim, smemSize, streams[0]>>>(dA, dB, dCm, num);

  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hCm, dCm, size, cudaMemcpyDeviceToHost);
  int ret1 = verify(hA, hB, hC, num, 1);
  if (ret1 != 0) {
    printf("Error: vectorAdd test result verification failed!\n");
  } else {
    printf("vectorAdd test result verification successed!\n");
  }

  int ret2 = verify(hA, hB, hCm, num, 0);
  if (ret2 != 0) {
    printf("Error: vectorMul test result verification failed!\n");
  } else {
    printf("vectorMul test result verification successed!\n");
  }
  

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dCm);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  free(hA);
  free(hB);
  free(hC);
  free(hCm);
  EXPECT_EQ(ret1|ret2, 0);
}



#include <assert.h>
#include <stdio.h>


#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif


__global__ void simplePrintfKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",
           blockIdx.y * gridDim.x + blockIdx.x,                                            
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x, 
           val);
}

int main(int argc, char **argv)
{
    
    int devID = 0;
    cudaSetDevice(devID);

    
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);

    
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n", devID, major, minor, smCount);

    printf("printf() is called. Output:\n\n");

    
    
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    simplePrintfKernel<<<dimGrid, dimBlock>>>(10);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

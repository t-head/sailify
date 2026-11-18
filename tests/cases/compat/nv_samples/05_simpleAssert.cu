


#include <cassert>
#include <stdio.h>


#include <cuda_runtime.h>

const char *sampleName = "simpleAssert";


bool testResult = true;


__global__ void simpleAssertKernel(int N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < N);
}

int main(int argc, char **argv)
{
    int         Nblocks  = 2;
    int         Nthreads = 32;
    cudaError_t error;

    printf("%s starting...\n\n", sampleName);

    
    int devID = 0;
    cudaSetDevice(devID);

    
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);

    
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n", devID, major, minor, smCount);

    
    
    dim3 dimGrid(Nblocks);
    dim3 dimBlock(Nthreads);

    printf("Launch kernel to generate assertion failures\n");
    simpleAssertKernel<<<dimGrid, dimBlock>>>(60);

    
    printf("\n-- Begin assert output\n\n");
    error = cudaDeviceSynchronize();
    printf("\n-- End assert output\n\n");

    
    if (error == cudaErrorAssert) {
        printf("Device assert failed as expected, "
               "CUDA error message is: %s\n\n",
               cudaGetErrorString(error));
    }

    testResult = error == cudaErrorAssert;

    printf("%s completed, returned %s\n", sampleName, testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

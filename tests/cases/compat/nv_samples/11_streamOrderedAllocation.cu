


#include <assert.h>
#include <climits>
#include <stdio.h>


#include <cuda_runtime.h>


#include <helper_cuda.h>
#include <helper_functions.h>

#define MAX_ITER 20


__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int basicStreamOrderedAllocation(const int dev, const int nelem, const float *a, const float *b, float *c)
{
    float *d_a, *d_b, *d_c; 
    float  errorNorm, refNorm, ref, diff;
    size_t bytes = nelem * sizeof(float);

    cudaStream_t stream;
    printf("Starting basicStreamOrderedAllocation()\n");
    checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
    checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

    checkCudaErrors(cudaFreeAsync(d_a, stream));
    checkCudaErrors(cudaFreeAsync(d_b, stream));
    checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaFreeAsync(d_c, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm   = 0.f;

    for (int n = 0; n < nelem; n++) {
        ref  = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm   = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
        printf("basicStreamOrderedAllocation PASSED\n");

    checkCudaErrors(cudaStreamDestroy(stream));

    return errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}


int streamOrderedAllocationPostSync(const int dev, const int nelem, const float *a, const float *b, float *c)
{
    float *d_a, *d_b, *d_c; 
    float  errorNorm, refNorm, ref, diff;
    size_t bytes = nelem * sizeof(float);

    cudaStream_t  stream;
    cudaMemPool_t memPool;
    cudaEvent_t   start, end;
    printf("Starting streamOrderedAllocationPostSync()\n");
    checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&memPool, dev));
    uint64_t thresholdVal = ULONG_MAX;
    
    
    
    
    
    checkCudaErrors(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal));

    
    checkCudaErrors(cudaEventRecord(start, stream));
    for (int i = 0; i < MAX_ITER; i++) {
        checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
        checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

        dim3 block(256);
        dim3 grid((unsigned int)ceil(nelem / (float)block.x));
        vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

        checkCudaErrors(cudaFreeAsync(d_a, stream));
        checkCudaErrors(cudaFreeAsync(d_b, stream));
        checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaFreeAsync(d_c, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    checkCudaErrors(cudaEventRecord(end, stream));
    
    checkCudaErrors(cudaEventSynchronize(end));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, end));
    printf("Total elapsed time = %f ms over %d iterations\n", msecTotal, MAX_ITER);

    
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm   = 0.f;

    for (int n = 0; n < nelem; n++) {
        ref  = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm   = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
        printf("streamOrderedAllocationPostSync PASSED\n");

    checkCudaErrors(cudaStreamDestroy(stream));

    return errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    int    nelem;
    int    dev = 0; 
    size_t bytes;
    float *a, *b, *c; 

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Usage:  streamOrderedAllocation [OPTION]\n\n");
        printf("Options:\n");
        printf("  --device=[device #]  Specify the device to be used\n");
        return EXIT_SUCCESS;
    }

    dev = findCudaDevice(argc, (const char **)argv);

    int isMemPoolSupported = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, dev));
    if (!isMemPoolSupported) {
        printf("Waiving execution as device does not support Memory Pools\n");
        exit(EXIT_WAIVED);
    }

    
    nelem = 1048576;
    bytes = nelem * sizeof(float);

    a = (float *)malloc(bytes);
    b = (float *)malloc(bytes);
    c = (float *)malloc(bytes);
    
    for (int n = 0; n < nelem; n++) {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    int ret1 = basicStreamOrderedAllocation(dev, nelem, a, b, c);
    int ret2 = streamOrderedAllocationPostSync(dev, nelem, a, b, c);

    
    free(a);
    free(b);
    free(c);

    return ((ret1 == EXIT_SUCCESS && ret2 == EXIT_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE);
}



#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>


#include <helper_functions.h> 


#include <helper_cuda.h> 

namespace cg = cooperative_groups;

#if __CUDA_ARCH__ >= 700
#endif

__global__ void normVecByDotProductAWBarrier(float *vecA, float *vecB, double *partialResults, int size)
{
#if __CUDA_ARCH__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
    cg::thread_block cta  = cg::this_thread_block();
    cg::grid_group   grid = cg::this_grid();
    ;


    if (threadIdx.x == 0) {
    }

    cg::sync(cta);

    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    }

    
    

    cg::sync(grid);

    
    
    if (blockIdx.x == 0) {
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
        }
    }

    cg::sync(grid);

    const double finalValue = partialResults[0];

    
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        vecA[i] = (float)vecA[i] / finalValue;
        vecB[i] = (float)vecB[i] / finalValue;
    }
#endif
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId);


int main(int argc, char **argv)
{
    printf("%s starting...\n", argv[0]);

    
    int dev = findCudaDevice(argc, (const char **)argv);

    int major = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

    
    if (major < 7) {
        printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    int supportsCooperativeLaunch = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&supportsCooperativeLaunch, cudaDevAttrCooperativeLaunch, dev));

    if (!supportsCooperativeLaunch) {
        printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
               "Waiving the run\n",
               dev);
        exit(EXIT_WAIVED);
    }

    int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

    printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId)
{
    float  *vecA, *d_vecA;
    float  *vecB, *d_vecB;
    double *d_partialResults;
    int     size = 10000000;

    checkCudaErrors(cudaMallocHost(&vecA, sizeof(float) * size));
    checkCudaErrors(cudaMallocHost(&vecB, sizeof(float) * size));

    checkCudaErrors(cudaMalloc(&d_vecA, sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_vecB, sizeof(float) * size));

    float baseVal = 2.0;
    for (int i = 0; i < size; i++) {
        vecA[i] = vecB[i] = baseVal;
    }

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMemcpyAsync(d_vecA, vecA, sizeof(float) * size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_vecB, vecB, sizeof(float) * size, cudaMemcpyHostToDevice, stream));

    
    
    int minGridSize = 0, blockSize = 0;
    checkCudaErrors(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)normVecByDotProductAWBarrier, 0, size));

    int smemSize = ((blockSize / 32) + 1) * sizeof(double);

    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

    int multiProcessorCount = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

    minGridSize = multiProcessorCount * numBlocksPerSm;
    checkCudaErrors(cudaMalloc(&d_partialResults, minGridSize * sizeof(double)));

    printf("Launching normVecByDotProductAWBarrier kernel with numBlocks = %d "
           "blockSize = %d\n",
           minGridSize,
           blockSize);

    dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

    void *kernelArgs[] = {(void *)&d_vecA, (void *)&d_vecB, (void *)&d_partialResults, (void *)&size};

    checkCudaErrors(cudaLaunchCooperativeKernel(
        (void *)normVecByDotProductAWBarrier, dimGrid, dimBlock, kernelArgs, smemSize, stream));

    checkCudaErrors(cudaMemcpyAsync(vecA, d_vecA, sizeof(float) * size, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    float        expectedResult = (baseVal / sqrt(size * baseVal * baseVal));
    unsigned int matches        = 0;
    for (int i = 0; i < size; i++) {
        if ((vecA[i] - expectedResult) > 0.00001) {
            printf("mismatch at i = %d\n", i);
            break;
        }
        else {
            matches++;
        }
    }

    printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
    checkCudaErrors(cudaFree(d_vecA));
    checkCudaErrors(cudaFree(d_vecB));
    checkCudaErrors(cudaFree(d_partialResults));

    checkCudaErrors(cudaFreeHost(vecA));
    checkCudaErrors(cudaFreeHost(vecB));
    return matches == size;
}

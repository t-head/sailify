


#include <assert.h>
#include <stdio.h>


#include <cuda_runtime.h>


#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

static const char *sSDKsample = "[simpleVoteIntrinsics]\0";


#define VOTE_DATA_GROUP 4


#ifndef SIMPLEVOTE_KERNEL_CU
#define SIMPLEVOTE_KERNEL_CU


__global__ void VoteAnyKernel1(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask   = 0xffffffff;
    result[tx] = __any_sync(mask, input[tx]);
}


__global__ void VoteAllKernel2(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask   = 0xffffffff;
    result[tx] = __all_sync(mask, input[tx]);
}


__global__ void VoteAnyKernel3(bool *info, int warp_size)
{
    int          tx   = threadIdx.x;
    unsigned int mask = 0xffffffff;
    bool        *offs = info + (tx * 3);

    
    *offs = __any_sync(mask, (tx >= (warp_size * 3) / 2));
    
    
    *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);

    
    if (__all_sync(mask, (tx >= (warp_size * 3) / 2))) {
        *(offs + 2) = true;
    }
}

#endif


void genVoteTestPattern(unsigned int *VOTE_PATTERN, int size)
{
    
    for (int i = 0; i < size / 4; i++) {
        VOTE_PATTERN[i] = 0x00000000;
    }

    
    for (int i = 2 * size / 8; i < 4 * size / 8; i++) {
        VOTE_PATTERN[i] = (i & 0x01) ? i : 0;
    }

    
    for (int i = 2 * size / 4; i < 3 * size / 4; i++) {
        VOTE_PATTERN[i] = (i & 0x01) ? 0 : i;
    }

    
    for (int i = 3 * size / 4; i < 4 * size / 4; i++) {
        VOTE_PATTERN[i] = 0xffffffff;
    }
}

int checkErrors1(unsigned int *h_result, int start, int end, int warp_size, const char *voteType)
{
    int i, sum = 0;

    for (sum = 0, i = start; i < end; i++) {
        sum += h_result[i];
    }

    if (sum > 0) {
        printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

        for (i = start; i < end; i++) {
            printf("%d", h_result[i]);
        }

        printf("%d values FAILED\n", sum);
    }

    return (sum > 0);
}

int checkErrors2(unsigned int *h_result, int start, int end, int warp_size, const char *voteType)
{
    int i, sum = 0;

    for (sum = 0, i = start; i < end; i++) {
        sum += h_result[i];
    }

    if (sum != warp_size) {
        printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

        for (i = start; i < end; i++) {
            printf("%d", h_result[i]);
        }

        printf(" - FAILED\n");
    }

    return (sum != warp_size);
}


int checkResultsVoteAnyKernel1(unsigned int *h_result, int size, int warp_size)
{
    int error_count = 0;

    error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
    error_count += checkErrors2(
        h_result, VOTE_DATA_GROUP * warp_size / 4, 2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
    error_count += checkErrors2(
        h_result, 2 * VOTE_DATA_GROUP * warp_size / 4, 3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
    error_count += checkErrors2(
        h_result, 3 * VOTE_DATA_GROUP * warp_size / 4, 4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");

    printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
    return error_count;
}


int checkResultsVoteAllKernel2(unsigned int *h_result, int size, int warp_size)
{
    int error_count = 0;

    error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
    error_count += checkErrors1(
        h_result, VOTE_DATA_GROUP * warp_size / 4, 2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
    error_count += checkErrors1(
        h_result, 2 * VOTE_DATA_GROUP * warp_size / 4, 3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
    error_count += checkErrors2(
        h_result, 3 * VOTE_DATA_GROUP * warp_size / 4, 4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");

    printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
    return error_count;
}


int checkResultsVoteAnyKernel3(bool *hinfo, int size)
{
    int i, error_count = 0;

    for (i = 0; i < size * 3; i++) {
        switch (i % 3) {
        case 0:

            
            if (hinfo[i] != (i >= size * 1)) {
                error_count++;
            }

            break;

        case 1:

            
            if (hinfo[i] != (i >= size * 3 / 2)) {
                error_count++;
            }

            break;

        case 2:

            
            if (hinfo[i] != (i >= size * 2)) {
                error_count++;
            }

            break;
        }
    }

    printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
    return error_count;
}

int main(int argc, char **argv)
{
    unsigned int *h_input, *h_result;
    unsigned int *d_input, *d_result;

    bool *dinfo = NULL, *hinfo = NULL;
    int   error_count[3] = {0, 0, 0};

    cudaDeviceProp deviceProp;
    int            devID, warp_size = 32;

    printf("%s\n", sSDKsample);

    
    devID = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount,
           deviceProp.major,
           deviceProp.minor);

    h_input  = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
    h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&d_input), VOTE_DATA_GROUP * warp_size * sizeof(unsigned int)));
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&d_result), VOTE_DATA_GROUP * warp_size * sizeof(unsigned int)));
    genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);
    checkCudaErrors(
        cudaMemcpy(d_input, h_input, VOTE_DATA_GROUP * warp_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    
    printf("[VOTE Kernel Test 1/3]\n");
    printf("\tRunning <<Vote.Any>> kernel1 ...\n");
    {
        checkCudaErrors(cudaDeviceSynchronize());
        dim3 gridBlock(1, 1);
        dim3 threadBlock(VOTE_DATA_GROUP * warp_size, 1);
        VoteAnyKernel1<<<gridBlock, threadBlock>>>(d_input, d_result, VOTE_DATA_GROUP * warp_size);
        getLastCudaError("VoteAnyKernel() execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(
        cudaMemcpy(h_result, d_result, VOTE_DATA_GROUP * warp_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    error_count[0] += checkResultsVoteAnyKernel1(h_result, VOTE_DATA_GROUP * warp_size, warp_size);

    
    printf("\n[VOTE Kernel Test 2/3]\n");
    printf("\tRunning <<Vote.All>> kernel2 ...\n");
    {
        checkCudaErrors(cudaDeviceSynchronize());
        dim3 gridBlock(1, 1);
        dim3 threadBlock(VOTE_DATA_GROUP * warp_size, 1);
        VoteAllKernel2<<<gridBlock, threadBlock>>>(d_input, d_result, VOTE_DATA_GROUP * warp_size);
        getLastCudaError("VoteAllKernel() execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(
        cudaMemcpy(h_result, d_result, VOTE_DATA_GROUP * warp_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    error_count[1] += checkResultsVoteAllKernel2(h_result, VOTE_DATA_GROUP * warp_size, warp_size);

    
    hinfo = reinterpret_cast<bool *>(calloc(warp_size * 3 * 3, sizeof(bool)));
    cudaMalloc(reinterpret_cast<void **>(&dinfo), warp_size * 3 * 3 * sizeof(bool));
    cudaMemcpy(dinfo, hinfo, warp_size * 3 * 3 * sizeof(bool), cudaMemcpyHostToDevice);

    printf("\n[VOTE Kernel Test 3/3]\n");
    printf("\tRunning <<Vote.Any>> kernel3 ...\n");
    {
        checkCudaErrors(cudaDeviceSynchronize());
        VoteAnyKernel3<<<1, warp_size * 3>>>(dinfo, warp_size);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cudaMemcpy(hinfo, dinfo, warp_size * 3 * 3 * sizeof(bool), cudaMemcpyDeviceToHost);

    error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

    
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_result));
    free(h_input);
    free(h_result);

    
    free(hinfo);
    cudaFree(dinfo);

    printf("\tShutting down...\n");

    return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

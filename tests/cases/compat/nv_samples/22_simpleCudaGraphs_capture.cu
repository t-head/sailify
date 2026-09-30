


#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define THREADS_PER_BLOCK 512


typedef struct callBackData
{
    const char *fn_name;
    double     *data;
} callBackData_t;


inline void CUDART_CB myHostNodeCallback(void *data)
{
    callBackData_t *tmp    = (callBackData_t *)(data);
    double         *result = (double *)(tmp->data);
    printf("[%s] Host callback final reduced sum = %lf\n", tmp->fn_name, *result);
    *result = 0.0; 
}


inline void init_input(float *inputVec, size_t size)
{
    for (size_t i = 0; i < size; i++)
        inputVec[i] = (rand() & 0xFF) / (float)RAND_MAX;
}


__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize, size_t outputSize)
{
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    
    double temp_sum = 0.0;
    for (size_t i = globaltid; i < inputSize; i += (size_t)gridDim.x * blockDim.x)
        temp_sum += (double)inputVec[i];

    
    double block_sum = BlockReduceT(temp_storage).Sum(temp_sum);

    if (threadIdx.x == 0 && blockIdx.x < outputSize)
        outputVec[blockIdx.x] = block_sum;
}


__global__ void reduceFinal(double *inputVec, double *result, size_t inputSize)
{
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    
    double temp_sum = 0.0;
    for (size_t i = globaltid; i < inputSize; i += (size_t)gridDim.x * blockDim.x)
        temp_sum += (double)inputVec[i];

    
    double block_sum = BlockReduceT(temp_storage).Sum(temp_sum);

    if (threadIdx.x == 0)
        result[0] = block_sum;
}


#include <cuda_runtime.h>
#include <cstdio>

#define GRAPH_LAUNCH_ITERATIONS 3

void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                                  float  *inputVec_d,
                                  double *outputVec_d,
                                  double *result_d,
                                  size_t  inputSize,
                                  size_t  numOfBlocks)
{
    cudaStream_t stream1, streamForGraph;
    cudaGraph_t  graph;
    double       result_h = 0.0;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&streamForGraph);

    
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
    cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

    callBackData_t hostFnData = {0};
    hostFnData.data           = &result_h;
    hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
    cudaLaunchHostFunc(stream1, myHostNodeCallback, &hostFnData);

    
    
    cudaStreamEndCapture(stream1, &graph);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, NULL, &numNodes);
    printf("Graph node count: %zu\n", numNodes);

    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    
    
    cudaGraph_t     clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    cudaGraphClone(&clonedGraph, graph);
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);

    
    
    
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    printf("\nCloned graph:\n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(clonedGraphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(streamForGraph);
}

int main()
{
    size_t size      = 1 << 24; 
    size_t maxBlocks = 512;

    int devID = 0;
    cudaSetDevice(devID);

    int major, minor, smCount;
    cudaDeviceGetAttribute(&major,   cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor,   cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,    devID);
    printf("GPU Device %d: compute capability %d.%d, %d SMs\n\n", devID, major, minor, smCount);

    printf("Reducing %zu elements\n", size);
    printf("Threads per block   : %d\n", THREADS_PER_BLOCK);
    printf("Graph launch iterations: %d\n\n", GRAPH_LAUNCH_ITERATIONS);

    float  *inputVec_h = NULL, *inputVec_d = NULL;
    double *outputVec_d = NULL, *result_d = NULL;

    cudaMallocHost(&inputVec_h, sizeof(float) * size);
    cudaMalloc(&inputVec_d, sizeof(float) * size);
    cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks);
    cudaMalloc(&result_d, sizeof(double));

    printf("=== Stream Capture ===\n");
    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
    cudaFreeHost(inputVec_h);
    return EXIT_SUCCESS;
}

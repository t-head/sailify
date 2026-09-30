

#include <cuda_runtime.h>




#define NUM_THREADS 1000000
#define BLOCK_WIDTH 1000
#define ARRAY_SIZE  10


__global__ void increment(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    if (tid < NUM_THREADS) {
        
        int i = tid % ARRAY_SIZE;
        g[i]  = g[i] + 1;
    }
}


__global__ void increment_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        atomicAdd(&g[i], 1);
    }
}


__global__ void max(int *g)
{
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        if (g[i] < tid)
            g[i] = tid;
    }
}


__global__ void max_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        atomicMax(&g[i], tid);
    }
}


__global__ void cas(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        int expected = g[i];        
        if (g[i] == expected)       
            g[i] = expected + 1;    
    }
}


__global__ void cas_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        int old = g[i];
        int assumed;
        do {
            assumed = old;
            old     = atomicCAS(&g[i], assumed, assumed + 1);
        } while (old != assumed);
    }
}


void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("}\n");
}

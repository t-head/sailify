#include <cuda_runtime.h>


#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;


__device__ int sumReduction(thread_group g, int *x, int val)
{
    
    int lane = g.thread_rank();

    
    
    
    
    for (int i = g.size() / 2; i > 0; i /= 2) {
        
        x[lane] = val;

        
        g.sync();

        if (lane < i)
            
            
            val += x[lane + i];

        
        g.sync();
    }

    
    if (g.thread_rank() == 0)
        return val;
    else
        return -1;
}


__global__ void cgkernel()
{
    
    thread_block threadBlockGroup     = this_thread_block();
    int          threadBlockGroupSize = threadBlockGroup.size();

    
    extern __shared__ int workspace[];

    int input, output, expectedOutput;

    
    input = threadBlockGroup.thread_rank();

    
    
    expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

    
    output = sumReduction(threadBlockGroup, workspace, input);

    
    if (threadBlockGroup.thread_rank() == 0) {
        printf(" Sum of all ranks 0..%d in threadBlockGroup is %d (expected %d)\n\n",
               (int)threadBlockGroup.size() - 1,
               output,
               expectedOutput);

        printf(" Now creating %d groups, each of size 16 threads:\n\n", (int)threadBlockGroup.size() / 16);
    }

    threadBlockGroup.sync();

    
    thread_block_tile<16> tiledPartition16 = tiled_partition<16>(threadBlockGroup);

    
    
    int workspaceOffset = threadBlockGroup.thread_rank() - tiledPartition16.thread_rank();

    
    input = tiledPartition16.thread_rank();

    
    
    expectedOutput = 15 * 16 / 2;

    
    output = sumReduction(tiledPartition16, workspace + workspaceOffset, input);

    
    if (tiledPartition16.thread_rank() == 0)
        printf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d "
               "(expected %d)\n",
               output,
               expectedOutput);

    return;
}


int main()
{
    
    cudaError_t err;

    

    int blocksPerGrid   = 1;
    int threadsPerBlock = 64;

    printf("\nLaunching a single block with %d threads...\n\n", threadsPerBlock);

    
    
    cgkernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>();
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\n...Done.\n\n");

    return 0;
}

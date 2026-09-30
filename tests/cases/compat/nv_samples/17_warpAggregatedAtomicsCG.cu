


#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

namespace cg = cooperative_groups;

#define NUM_ELEMS             10000000
#define NUM_THREADS_PER_BLOCK 512


__device__ int atomicAggInc(int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();

    
    int res = 0;
    if (active.thread_rank() == 0) {
        res = atomicAdd(counter, active.size());
    }

    
    res = active.shfl(res, 0);

    
    return res + active.thread_rank();
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = id; i < n; i += gridDim.x * blockDim.x) {
        if (src[i] > 0)
            dst[atomicAggInc(nres)] = src[i];
    }
}


#if __CUDA_ARCH__ >= 700
__device__ int atomicAggIncMulti(const int bucket, int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();
    
    auto labeledGroup = cg::labeled_partition(active, bucket);

    int res = 0;
    if (labeledGroup.thread_rank() == 0) {
        res = atomicAdd(&counter[bucket], labeledGroup.size());
    }

    
    res = labeledGroup.shfl(res, 0);

    
    return res + labeledGroup.thread_rank();
}
#endif


__global__ void
mapToBuckets(const int *srcArr, int *indicesBuckets, int *bucketCounters, const int srcSize, const int numOfBuckets)
{
#if __CUDA_ARCH__ >= 700
    cg::grid_group grid = cg::this_grid();

    for (int i = grid.thread_rank(); i < srcSize; i += grid.size()) {
        const int bucket = srcArr[i];
        if (bucket < numOfBuckets) {
            indicesBuckets[atomicAggIncMulti(bucket, bucketCounters)] = i;
        }
    }
#endif
}

int mapIndicesToBuckets(int *h_srcArr, int *d_srcArr, int numOfBuckets)
{
    int *d_indicesBuckets, *d_bucketCounters;
    int *cpuBucketCounters = new int[numOfBuckets];
    int *h_bucketCounters  = new int[numOfBuckets];

    memset(cpuBucketCounters, 0, sizeof(int) * numOfBuckets);
    
    for (int i = 0; i < numOfBuckets; i++) {
        h_bucketCounters[i] = i * NUM_ELEMS;
    }

    checkCudaErrors(cudaMalloc(&d_indicesBuckets, sizeof(int) * NUM_ELEMS * numOfBuckets));
    checkCudaErrors(cudaMalloc(&d_bucketCounters, sizeof(int) * numOfBuckets));

    checkCudaErrors(cudaMemcpy(d_bucketCounters, h_bucketCounters, sizeof(int) * numOfBuckets, cudaMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK), 1, 1);

    mapToBuckets<<<dimGrid, dimBlock>>>(d_srcArr, d_indicesBuckets, d_bucketCounters, NUM_ELEMS, numOfBuckets);

    checkCudaErrors(cudaMemcpy(h_bucketCounters, d_bucketCounters, sizeof(int) * numOfBuckets, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_ELEMS; i++) {
        cpuBucketCounters[h_srcArr[i]]++;
    }

    bool allMatch   = true;
    int  finalElems = 0;
    for (int i = 0; i < numOfBuckets; i++) {
        finalElems += (h_bucketCounters[i] - i * NUM_ELEMS);
        if (cpuBucketCounters[i] != (h_bucketCounters[i] - i * NUM_ELEMS)) {
            allMatch = false;
            break;
        }
    }

    if (!allMatch && finalElems != NUM_ELEMS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


#if __CUDA_ARCH__ >= 700
__device__ void atomicAggMaxMulti(const int bucket, int *counter, const int valueForMax)
{
    cg::coalesced_group active = cg::coalesced_threads();
    
    auto labeledGroup = cg::labeled_partition(active, bucket);

    const int maxValueInGroup = cg::reduce(labeledGroup, valueForMax, cg::greater<int>());

    if (labeledGroup.thread_rank() == 0) {
        atomicMax(&counter[bucket], maxValueInGroup);
    }
}
#endif


__global__ void calculateMaxInEachBuckets(const int *srcArr,
                                          const int *valueInBuckets,
                                          int       *bucketsMax,
                                          const int  srcSize,
                                          const int  numOfBuckets)
{
#if __CUDA_ARCH__ >= 700
    cg::grid_group grid = cg::this_grid();

    for (int i = grid.thread_rank(); i < srcSize; i += grid.size()) {
        const int bucket = srcArr[i];
        if (bucket < numOfBuckets) {
            atomicAggMaxMulti(bucket, bucketsMax, valueInBuckets[i]);
        }
    }
#endif
}

int calculateMaxInBuckets(int *h_srcArr, int *d_srcArr, int numOfBuckets)
{
    int *d_valueInBuckets, *d_bucketsMax;
    int *h_valueInBuckets = new int[NUM_ELEMS];
    int *cpuBucketsMax    = new int[numOfBuckets];
    int *h_bucketsMax     = new int[numOfBuckets];

    memset(cpuBucketsMax, 0, sizeof(int) * numOfBuckets);

    
    
    for (int i = 0; i < NUM_ELEMS; i++) {
        h_valueInBuckets[i] = rand();
    }

    checkCudaErrors(cudaMalloc(&d_valueInBuckets, sizeof(int) * NUM_ELEMS));
    checkCudaErrors(cudaMalloc(&d_bucketsMax, sizeof(int) * numOfBuckets));

    checkCudaErrors(cudaMemset(d_bucketsMax, 0, sizeof(int) * numOfBuckets));
    checkCudaErrors(cudaMemcpy(d_valueInBuckets, h_valueInBuckets, sizeof(int) * NUM_ELEMS, cudaMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK), 1, 1);

    calculateMaxInEachBuckets<<<dimGrid, dimBlock>>>(d_srcArr, d_valueInBuckets, d_bucketsMax, NUM_ELEMS, numOfBuckets);

    checkCudaErrors(cudaMemcpy(h_bucketsMax, d_bucketsMax, sizeof(int) * numOfBuckets, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_ELEMS; i++) {
        if (cpuBucketsMax[h_srcArr[i]] < h_valueInBuckets[i]) {
            cpuBucketsMax[h_srcArr[i]] = h_valueInBuckets[i];
        }
    }

    bool allMatch   = true;
    int  finalElems = 0;
    for (int i = 0; i < numOfBuckets; i++) {
        if (cpuBucketsMax[i] != h_bucketsMax[i]) {
            allMatch = false;
            printf("CPU i=%d  max = %d mismatches GPU max = %d\n", i, cpuBucketsMax[i], h_bucketsMax[i]);
            break;
        }
    }
    if (allMatch) {
        printf("CPU max matches GPU max\n");
    }

    delete[] h_valueInBuckets;
    delete[] cpuBucketsMax;
    delete[] h_bucketsMax;
    checkCudaErrors(cudaFree(d_valueInBuckets));
    checkCudaErrors(cudaFree(d_bucketsMax));

    if (!allMatch && finalElems != NUM_ELEMS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    int *data_to_filter, *filtered_data, nres = 0;
    int *d_data_to_filter, *d_filtered_data, *d_nres;

    int numOfBuckets = 5;

    data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

    
    for (int i = 0; i < NUM_ELEMS; i++) {
        data_to_filter[i] = rand() % numOfBuckets;
    }

    int devId = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaMalloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS));
    checkCudaErrors(cudaMalloc(&d_filtered_data, sizeof(int) * NUM_ELEMS));
    checkCudaErrors(cudaMalloc(&d_nres, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_data_to_filter, data_to_filter, sizeof(int) * NUM_ELEMS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_nres, 0, sizeof(int)));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK) + 1, 1, 1);

    filter_arr<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter, NUM_ELEMS);

    checkCudaErrors(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));

    filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

    checkCudaErrors(cudaMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres, cudaMemcpyDeviceToHost));

    int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

    
    int host_flt_count = 0;
    for (int i = 0; i < NUM_ELEMS; i++) {
        if (data_to_filter[i] > 0) {
            host_filtered_data[host_flt_count++] = data_to_filter[i];
        }
    }

    int major = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devId));

    int mapIndicesToBucketsStatus   = EXIT_SUCCESS;
    int calculateMaxInBucketsStatus = EXIT_SUCCESS;
    
    
    if (major >= 7) {
        mapIndicesToBucketsStatus   = mapIndicesToBuckets(data_to_filter, d_data_to_filter, numOfBuckets);
        calculateMaxInBucketsStatus = calculateMaxInBuckets(data_to_filter, d_data_to_filter, numOfBuckets);
    }

    printf("\nWarp Aggregated Atomics %s \n",
           (host_flt_count == nres) && (mapIndicesToBucketsStatus == EXIT_SUCCESS)
                   && (calculateMaxInBucketsStatus == EXIT_SUCCESS)
               ? "PASSED"
               : "FAILED");

    checkCudaErrors(cudaFree(d_data_to_filter));
    checkCudaErrors(cudaFree(d_filtered_data));
    checkCudaErrors(cudaFree(d_nres));
    free(data_to_filter);
    free(filtered_data);
    free(host_filtered_data);
}

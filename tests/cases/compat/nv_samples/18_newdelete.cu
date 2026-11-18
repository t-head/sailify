#include <cuda_runtime.h>


#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;
#include <algorithm>
#include <helper_cuda.h>
#include <stdlib.h>
#include <vector>

const char *sSDKsample = "newdelete";


template <class T> class Container
{
public:
    __device__ Container() { ; }

    __device__ virtual ~Container() { ; }

    __device__ virtual void push(T e) = 0;

    __device__ virtual bool pop(T &e) = 0;
};


template <class T> class Vector : public Container<T>
{
public:
    
    
    __device__ Vector(int max_size)
        : m_top(-1)
    {
        m_data = new T[max_size];
    }

    
    __device__ Vector(int max_size, T *preallocated_buffer)
        : m_top(-1)
    {
        m_data = new (preallocated_buffer) T[max_size];
    }

    
    
    __device__ ~Vector()
    {
        if (m_data)
            delete[] m_data;
    }

    __device__ virtual void push(T e)
    {
        if (m_data) {
            
            int idx         = atomicAdd(&(this->m_top), 1);
            m_data[idx + 1] = e;
        }
    }

    __device__ virtual bool pop(T &e)
    {
        if (m_data && m_top >= 0) {
            
            int idx = atomicAdd(&(this->m_top), -1);
            if (idx >= 0) {
                e = m_data[idx];
                return true;
            }
        }
        return false;
    }

private:
    int m_size;
    T  *m_data;

    int m_top;
};


__global__ void vectorCreate(Container<int> **g_container, int max_size)
{
    
    
    

    *g_container = new Vector<int>(max_size);
}


__global__ void containerFill(Container<int> **g_container)
{
    
    
    if (threadIdx.x == 0) {
        (*g_container)->push(blockIdx.x);
    }
}

__global__ void containerConsume(Container<int> **g_container, int *d_result)
{
    
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int v;

    if ((*g_container)->pop(v)) {
        d_result[idx] = v;
    }
    else {
        d_result[idx] = -1;
    }
}


__global__ void containerDelete(Container<int> **g_container) { delete *g_container; }


__global__ void placementNew(int *d_result)
{
    
    cg::thread_block         cta = cg::this_thread_block();
    __shared__ unsigned char __align__(8) s_buffer[sizeof(Vector<int>)];
    __shared__ int           __align__(8) s_data[1024];
    __shared__ Vector<int> *s_vector;

    
    
    
    if (threadIdx.x == 0) {
        s_vector = new (s_buffer) Vector<int>(1024, s_data);
    }

    cg::sync(cta);

    if ((threadIdx.x & 1) == 0) {
        s_vector->push(threadIdx.x >> 1);
    }

    
    
    cg::sync(cta);

    int v;

    if (s_vector->pop(v)) {
        d_result[threadIdx.x] = v;
    }
    else {
        d_result[threadIdx.x] = -1;
    }

    
    
}

struct ComplexType_t
{
    int   a;
    int   b;
    float c;
    float d;
};

__global__ void complexVector(int *d_result)
{
    
    cg::thread_block         cta = cg::this_thread_block();
    __shared__ unsigned char __align__(8) s_buffer[sizeof(Vector<ComplexType_t>)];
    __shared__ ComplexType_t __align__(8) s_data[1024];
    __shared__ Vector<ComplexType_t> *s_vector;

    
    
    
    if (threadIdx.x == 0) {
        s_vector = new (s_buffer) Vector<ComplexType_t>(1024, s_data);
    }

    cg::sync(cta);

    if ((threadIdx.x & 1) == 0) {
        ComplexType_t data;
        data.a = threadIdx.x >> 1;
        data.b = blockIdx.x;
        data.c = threadIdx.x / (float)(blockDim.x);
        data.d = blockIdx.x / (float)(gridDim.x);

        s_vector->push(data);
    }

    cg::sync(cta);

    ComplexType_t v;

    if (s_vector->pop(v)) {
        d_result[threadIdx.x] = v.a;
    }
    else {
        d_result[threadIdx.x] = -1;
    }

    
    
}


bool checkResult(int *d_result, int N)
{
    std::vector<int> h_result;
    h_result.resize(N);

    checkCudaErrors(cudaMemcpy(&h_result[0], d_result, N * sizeof(int), cudaMemcpyDeviceToHost));
    std::sort(h_result.begin(), h_result.end());

    bool success = true;
    bool test    = false;

    int value = 0;

    for (int i = 0; i < N; ++i) {
        if (h_result[i] != -1) {
            test = true;
        }

        if (test && (value++) != h_result[i]) {
            success = false;
        }
    }

    return success;
}

bool testContainer(Container<int> **d_container, int blocks, int threads)
{
    int *d_result;
    cudaMalloc(&d_result, blocks * threads * sizeof(int));

    containerFill<<<blocks, threads>>>(d_container);
    containerConsume<<<blocks, threads>>>(d_container, d_result);
    containerDelete<<<1, 1>>>(d_container);
    checkCudaErrors(cudaDeviceSynchronize());

    bool success = checkResult(d_result, blocks * threads);

    cudaFree(d_result);

    return success;
}

bool testPlacementNew(int threads)
{
    int *d_result;
    cudaMalloc(&d_result, threads * sizeof(int));

    placementNew<<<1, threads>>>(d_result);
    checkCudaErrors(cudaDeviceSynchronize());

    bool success = checkResult(d_result, threads);

    cudaFree(d_result);

    return success;
}

bool testComplexType(int threads)
{
    int *d_result;
    cudaMalloc(&d_result, threads * sizeof(int));

    complexVector<<<1, threads>>>(d_result);
    checkCudaErrors(cudaDeviceSynchronize());

    bool success = checkResult(d_result, threads);

    cudaFree(d_result);

    return success;
}


int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", sSDKsample);

    
    
    findCudaDevice(argc, (const char **)argv);

    
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * (1 << 20)));

    Container<int> **d_container;
    checkCudaErrors(cudaMalloc(&d_container, sizeof(Container<int> **)));

    bool bTest       = false;
    int  test_passed = 0;

    printf(" > Container = Vector test ");
    vectorCreate<<<1, 1>>>(d_container, 128 * 128);
    bTest = testContainer(d_container, 128, 128);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    checkCudaErrors(cudaFree(d_container));

    printf(" > Container = Vector, using placement new on SMEM buffer test ");
    bTest = testPlacementNew(1024);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    printf(" > Container = Vector, with user defined datatype test ");
    bTest = testComplexType(1024);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    printf("Test Summary: %d/3 succesfully run\n", test_passed);

    exit(test_passed == 3 ? EXIT_SUCCESS : EXIT_FAILURE);
}

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel(float* out) {
    out[threadIdx.x] = (float)threadIdx.x;
}

int main() {
    float* d = nullptr;
    cudaError_t err = cudaMalloc(&d, 32 * sizeof(float));
    if (err == cudaSuccess) {
        hello_kernel<<<2, 32>>>(d);
        cudaDeviceSynchronize();
        cudaFree(d);
        puts("hello kernel ok");
        return 0;
    }
    puts("cudaMalloc failed");
    return 1;
}

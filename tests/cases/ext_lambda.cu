#include <cstdio>
#include <cuda_runtime.h>

__global__ void apply_scale(float* v, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = v[i] * s;
    }
}

int main() {
    int n = 128;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    auto host_dev = [] __device__ (float x) { return x * 2.0f; };
    (void)host_dev;
    apply_scale<<<2, 64>>>(d, 1.5f, n);
    cudaDeviceSynchronize();
    cudaFree(d);
    puts("ext lambda ok");
    return 0;
}

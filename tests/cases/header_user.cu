#include <cstdio>
#include <math_ops.cuh>

__global__ void use_scale(float* v, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = scale_add(v[i], 1.0f, s);
    }
}

int main() {
    int n = 64;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    use_scale<<<1, 64>>>(d, 2.0f, n);
    cudaDeviceSynchronize();
    cudaFree(d);
    puts("header user ok");
    return 0;
}

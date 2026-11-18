#include "math_ops.cuh"

__global__ void scale_kernel(const float* in, float* out, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * s;
    }
}

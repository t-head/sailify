#include <cstdio>
#include <cuda_runtime.h>

__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (i < n) {
        v = in[i];
    }
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 512;
    float* din = nullptr;
    float* dout = nullptr;
    float* h = new float[n];
    for (int i = 0; i < n; i++) {
        h[i] = 1.0f;
    }
    cudaMalloc(&din, n * sizeof(float));
    cudaMalloc(&dout, 8 * sizeof(float));
    cudaMemcpy(din, h, n * sizeof(float), cudaMemcpyHostToDevice);
    block_sum<<<8, 64>>>(din, dout, n);
    cudaGetLastError();
    cudaFree(din);
    cudaFree(dout);
    delete[] h;
    puts("shared reduce ok");
    return 0;
}

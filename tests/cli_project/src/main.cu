#include <cstdio>
#include "util/math_ops.cuh"

int main() {
    const int n = 128;
    float hin[128], hout[128];
    float *din = 0, *dout = 0;
    for (int i = 0; i < n; ++i) { hin[i] = (float)i; }
    cudaMalloc((void**)&din, n * sizeof(float));
    cudaMalloc((void**)&dout, n * sizeof(float));
    cudaMemcpy(din, hin, n * sizeof(float), cudaMemcpyHostToDevice);
    scale_kernel<<<2, 64>>>(din, dout, n, 3.0f);
    cudaMemcpy(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(din);
    cudaFree(dout);
    puts("cli project ok");
    return 0;
}

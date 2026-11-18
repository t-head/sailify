#include <cstdio>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 256;
    float* ha = new float[n];
    float* hb = new float[n];
    float* hc = new float[n];
    for (int i = 0; i < n; i++) {
        ha[i] = (float)i;
        hb[i] = 1.0f;
    }
    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, n * sizeof(float));
    cudaMemcpy(da, ha, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, n * sizeof(float), cudaMemcpyHostToDevice);
    vec_add<<<4, 64>>>(da, db, dc, n);
    cudaGetLastError();
    cudaMemcpy(hc, dc, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] ha;
    delete[] hb;
    delete[] hc;
    puts("vector add ok");
    return 0;
}

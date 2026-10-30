#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <host_defines.h>
#include "crt/host_defines.h"
#include <channel_descriptor.h>
#include <surface_types.h>
#include <texture_types.h>
#include <cudaTypedefs.h>

__global__ void cat1_probe(float* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = v[i] + 1.0f;
    }
}

int main() {
    int n = 16;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cat1_probe<<<1, 16>>>(d, n);
    cudaDeviceSynchronize();
    cudaFree(d);
    puts("cat1 headers ok");
    return 0;
}

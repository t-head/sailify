#ifndef INCCHAIN_KERNEL_HOST_H
#define INCCHAIN_KERNEL_HOST_H
__global__ void chainKernel(int* out) {
    if (threadIdx.x == 0) {
        out[0] = 42;
    }
}
static constexpr int chainLaunchHelper(void) {
    return 7;
}
#endif

#ifndef UTIL_MATH_OPS_CUH
#define UTIL_MATH_OPS_CUH

__global__ void scale_kernel(const float* in, float* out, int n, float s);

#endif

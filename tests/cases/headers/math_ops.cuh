#ifndef MATH_OPS_CUH
#define MATH_OPS_CUH

__host__ __device__ inline float scale_add(float a, float b, float s) {
    return a + b * s;
}

#endif

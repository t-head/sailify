// cuda_fp4.h e2m1 API as used by sglang's deepseek_v4 JIT kernels and flashinfer.

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef __CUDA_FP4_TYPES_EXIST__
#error "cuda_fp4.h did not define __CUDA_FP4_TYPES_EXIST__"
#endif

static_assert(sizeof(__nv_fp4_e2m1) == 1, "fp4 e2m1 is one byte");
static_assert(sizeof(__nv_fp4x2_e2m1) == 1, "fp4x2 e2m1 is one byte");
static_assert(sizeof(__nv_fp4x4_e2m1) == 2, "fp4x4 e2m1 is two bytes");

// sglang deepseek_v4/kv_layout.cuh: store_row_fp4
__global__ void store_row_fp4(const float* v, float scale, uint8_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
        float2{__fdiv_rn(v[2 * i], scale), __fdiv_rn(v[2 * i + 1], scale)}, __NV_E2M1, cudaRoundNearest));
  }
}

// sglang deepseek_v4/fp4_utils.cuh: fake_quant_x2
__global__ void fake_quant_x2(const float2* x, float scale, float2* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const __nv_fp4x2_storage_t code = __nv_cvt_float2_to_fp4x2(x[i], __NV_E2M1, cudaRoundNearest);
    const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1);
    const float2 grid = __half22float2(__half2(raw));
    y[i] = float2{grid.x * scale, grid.y * scale};
  }
}

// flashinfer / trtllm style: struct storage and conversions
__global__ void struct_roundtrip(const float* x, const __half* h, __nv_fp4_e2m1* q, float* back, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    __nv_fp4_e2m1 a(x[i]);
    __nv_fp4_e2m1 b = __nv_fp4_e2m1(h[i]);
    q[i] = a;
    back[i] = static_cast<float>(a) + static_cast<float>(__half(b));
    __nv_fp4x2_e2m1 pair(float2{x[i], -x[i]});
    const float2 p = static_cast<float2>(pair);
    __nv_fp4x4_e2m1 quad(float4{x[i], p.x, p.y, 0.0f});
    back[i] += static_cast<float4>(quad).w;
  }
}

// Truncating mode and the remaining raw converters, callable from host too.
__host__ __device__ inline uint32_t pack_misc(double d, __nv_bfloat16_raw b, __half_raw h) {
  const __nv_fp4_storage_t s0 = __nv_cvt_double_to_fp4(d, __NV_E2M1, cudaRoundZero);
  const __nv_fp4_storage_t s1 = __nv_cvt_bfloat16raw_to_fp4(b, __NV_E2M1, cudaRoundNearest);
  const __nv_fp4_storage_t s2 = __nv_cvt_halfraw_to_fp4(h, __NV_E2M1, cudaRoundNearest);
  const __nv_fp4_storage_t s3 = __nv_cvt_float_to_fp4(1.25f, __NV_E2M1, cudaRoundZero);
  return s0 | (s1 << 4) | (s2 << 8) | (s3 << 12) | (__nv_cvt_fp4_to_halfraw(s0, __NV_E2M1).x << 16);
}

__global__ void misc(uint32_t* out, double d, __nv_bfloat16_raw b, __half_raw h) {
  out[threadIdx.x] = pack_misc(d, b, h);
}

uint32_t host_misc() {
  __nv_bfloat16_raw b;
  b.x = 0x3fc0;
  __half_raw h;
  h.x = 0x3e00;
  return pack_misc(2.5, b, h);
}

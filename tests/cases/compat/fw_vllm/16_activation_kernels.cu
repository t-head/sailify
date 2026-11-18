#include <cmath>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#else

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;#if defined(HIP_FP8_TYPE_OCP)
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3 __nv_fp8x4_e4m3;#else
                                             
typedef __hip_fp8_e4m3_fnuz __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3_fnuz __nv_fp8x4_e4m3;#endif
#endif

namespace vllm {

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
                   
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

__device__ __forceinline__ __nv_bfloat162 silu2_v2(float2 x) {
#ifndef USE_ROCM
  return make_bfloat162(__float2bfloat16_rn(silu(x.x)),
                        __float2bfloat16_rn(silu(x.y)));
#else
  return __float22bfloat162_rn(make_float2(silu(x.x), silu(x.y)));
#endif
}
#ifndef USE_ROCM
#endif

template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* _smem_ptr, const U* _glob_ptr) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  auto smem_ptr = reinterpret_cast<void*>(_smem_ptr);
  auto glob_ptr = reinterpret_cast<const void*>(_glob_ptr);
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
#else
  _smem_ptr[0] = _glob_ptr[0];
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#else
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#else
#endif
}

__device__ __forceinline__ float clip(float v, float mmin, float mmax) {
  return fminf(mmax, fmaxf(v, mmin));
}

__device__ __forceinline__ __nv_bfloat16 clip(__nv_bfloat16 v,
                                              __nv_bfloat16 mmin,
                                              __nv_bfloat16 mmax) {
  return __hmin(mmax, __hmax(v, mmin));
}

__device__ __forceinline__ __nv_bfloat162 clip(__nv_bfloat162 v,
                                               __nv_bfloat162 mmin,
                                               __nv_bfloat162 mmax) {
  return __hmin2(mmax, __hmax2(v, mmin));
}

template <typename Idx_t>
__device__ __forceinline__ int warp_expert_search(
    int idx, int n, const Idx_t* __restrict__ input, Idx_t val) {
  const Idx_t* input_ptr = input + idx;
  int base_offset = 0;

  for (;;) {
    bool move_on = (idx < n && *input_ptr <= val);

    unsigned mask = __ballot_sync(0xffffffff, move_on);

    if (mask != 0xffffffffu) {
      int last_lane = 31 - __clz(mask);
      return base_offset + last_lane;
    }

    input_ptr += 32;
    base_offset += 32;
    idx += 32;
  }
}

template <int num_parallel_tokens>
__device__ __forceinline__ void token_bounds(int32_t n_tokens,
                                             int32_t worker_id,
                                             int32_t& n_tokens_lower,
                                             int32_t& n_tokens_upper) {
  if (n_tokens < num_parallel_tokens && worker_id < n_tokens) {
    if (worker_id >= num_parallel_tokens) return;
    n_tokens_lower = worker_id;
    n_tokens_upper = worker_id + 1;
  } else {
    int32_t chunk_size = n_tokens / num_parallel_tokens;
    int32_t residual = n_tokens - chunk_size * num_parallel_tokens;
    auto calc_id = [&](int32_t id) {
      if (id < residual)
        return min(n_tokens, id * (chunk_size + 1));
      else
        return min(n_tokens, id * chunk_size + residual);
    };
    n_tokens_lower = calc_id(worker_id);
    n_tokens_upper = calc_id(worker_id + 1);
  }
}

}
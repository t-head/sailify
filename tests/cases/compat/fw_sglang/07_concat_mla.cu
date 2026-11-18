#include <cuda_runtime.h>
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#ifndef USE_MUSA
__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}
#else
constexpr int WarpSize = 32;
__forceinline__ __device__ int get_lane_id() {
  return threadIdx.x % WarpSize;
}
#endif

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ void st_na_global_v1(const int* ptr, int v) {
  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void st_na_global_v2(const int2* ptr, const int2& v) {
  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st_na_global_v4(const int4* ptr, const int4& v) {
  asm volatile(
      "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
      : "memory");
}

__device__ __forceinline__ int ld_na_global_v1(const int* ptr) {
  int r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ int2 ld_na_global_v2(const int2* ptr) {
  int2 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ int4 ld_na_global_v4(const int4* ptr) {
  int4 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ void prefetch_L2(const void* p) {
#if defined(ENABLE_L2_PREFETCH)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
}

                                                                        

constexpr int NUM_LOCAL_HEADS = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

constexpr int HEAD_CHUNK_SIZE = 16;
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;

__global__ void concat_mla_k_kernel(
    nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ k_nope,
    const nv_bfloat16* __restrict__ k_rope,
    const int num_tokens,
    const int64_t k_stride_0,
    const int k_stride_1,
    const int64_t k_nope_stride_0,
    const int k_nope_stride_1,
    const int64_t k_rope_stride_0) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  if (token_id >= num_tokens) return;

  using NopeVec = int2;                                   
  using RopeVec = int;                                    
  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), "rope vec mismatch");

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  const int2* __restrict__ nope_src =
      reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

  int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

  int* __restrict__ rope_dst =
      reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;

  const int nope_src_stride_v = (k_nope_stride_1 >> 2);                       
  const int nope_dst_stride_v = (k_stride_1 >> 2);
  const int rope_dst_stride_v = (k_stride_1 >> 1);                      

  const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);

  prefetch_L2(nope_src);
  NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      const int2* next_src = nope_src + nope_src_stride_v;
      prefetch_L2(next_src);
      next = ld_na_global_v2(next_src);
    }

    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);

    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;

    cur = next;
  }
}

                                                                                    

                                                          
constexpr int A_LAST_DIM = 512;
constexpr int B_LAST_DIM = 64;

__global__ void concat_mla_absorb_q_kernel(
    nv_bfloat16* a,
    nv_bfloat16* b,
    nv_bfloat16* out,
    const int num_items,
    const int dim_1,
    const int64_t a_stride_0,
    const int a_stride_1,
    const int64_t b_stride_0,
    const int b_stride_1,
    const int64_t out_stride_0,
    const int out_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = get_lane_id();

  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;

  if (flat_warp_id >= num_items) {
    return;
  }

  using ABufType = int4;
  constexpr int A_NUM_UNROLL = 2;
  static_assert(sizeof(ABufType) * A_NUM_UNROLL == A_LAST_DIM * sizeof(a[0]) / 32);
  ABufType a_buf[A_NUM_UNROLL];

  using BBufType = int;
  constexpr int B_NUM_UNROLL = 1;
  static_assert(sizeof(BBufType) * B_NUM_UNROLL == B_LAST_DIM * sizeof(b[0]) / 32);
  BBufType b_buf;

  {
    const BBufType* base_addr = reinterpret_cast<BBufType*>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
    b_buf = *(base_addr + lane_id);
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    const ABufType* base_addr = reinterpret_cast<ABufType*>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);
    a_buf[i] = *(base_addr + i * 32 + lane_id);
  }

  {
    BBufType* base_addr = reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + A_LAST_DIM);
    *(base_addr + lane_id) = b_buf;
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    ABufType* base_addr = reinterpret_cast<ABufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1);
    *(base_addr + i * 32 + lane_id) = a_buf[i];
  }
}
         

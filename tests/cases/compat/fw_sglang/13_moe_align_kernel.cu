#include <cuda_runtime.h>


#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

using namespace cute;


namespace flash {


template <typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template <>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
  asm volatile(
      "{\n"
      "\t .reg .f16x2 sela;\n"
      "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
      "\t and.b32 %0, sela, %1;\n"
      "}\n"
      : "=r"(res)
      : "r"(x), "r"(zero));
#endif
  return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
  asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
  return res;
}
#endif


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template <typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template <>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
  return res;
}

template <>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
  return res;
}

#endif


template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  
  __device__ __forceinline__ float operator()(float const& x, float const& y) {
    return max(x, y);
  }
};


template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x + y;
  }
};


template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};


template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};


}


#define VEC_SIZE 4
using Vec = int4;

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel,
    bool ignore_invalid_expert) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

#ifdef __CUDA_ARCH__
#endif

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  
  int32_t* prefix = shared_counts + num_experts;  
  int32_t* scan_buf = prefix + num_experts + 1;   
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

#ifndef __CUDA_ARCH__  

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  __syncthreads();

  
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

#else  

  
  int32_t* warp_sums = scan_buf + scan_size;  
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  __syncthreads();

  
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  
  if (tid < num_experts) prefix[tid] = scan_buf[tid];
#endif

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert,
    int32_t max_num_tokens_padded) {
  
  
  
  
  
  
  if (threadIdx.x < fill_threads) {
    
    if (pad_sorted_token_ids) {
      for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
        sorted_token_ids[it] = numel;
      }
    }
    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid - 1;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}


#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <flashinfer/vec_dtypes.cuh>
#include <iostream>

#include "cutlass/array.h"


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


constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(
    const int* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = static_cast<int32_t>(2 * n);
    problem_sizes1[expert_id * 3 + 2] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3 + 2] = static_cast<int32_t>(n);
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* blockscale_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  int32_t tot_rounded_offset = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    int num_tokens = problem_sizes1[i * 3];
    int rounded_num_tokens = (num_tokens + (128 - 1)) / 128 * 128;
    tot_offset += num_tokens;
    tot_rounded_offset += rounded_num_tokens;
    expert_offsets[i + 1] = tot_offset;
    blockscale_offsets[i + 1] = tot_rounded_offset;
  }
}

__global__ void compute_arg_sorts(
    const int32_t* __restrict__ topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  int expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}


template <typename T>
__global__ void shuffleRowsKernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    
    constexpr uint64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    
    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    auto const start_offset = threadIdx.x;
    auto const stride = blockDim.x;
    auto const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (auto elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

#define DECLARE_SHUFFLE_ROWS(T)      \
  __global__ void shuffleRowsKernel( \
      const T* input,                \
      const int32_t* dst2src_map,    \
      T* output,                     \
      int64_t num_src_rows,          \
      int64_t num_dest_rows,         \
      int64_t num_cols);

DECLARE_SHUFFLE_ROWS(float);
DECLARE_SHUFFLE_ROWS(half);
DECLARE_SHUFFLE_ROWS(__nv_bfloat16);
DECLARE_SHUFFLE_ROWS(uint8_t);

#define SHUFFLE_ROWS(T)                                    \
  shuffleRowsKernel<T><<<blocks, threads, 0, stream>>>(    \
      reinterpret_cast<const T*>(input),                   \
      static_cast<const int32_t*>(dst2src_map.data_ptr()), \
      reinterpret_cast<T*>(output),                        \
      num_src_rows,                                        \
      num_dst_rows,                                        \
      num_cols)

#define DTYPE_DISPATCH_CASE(T, CUDA_T) \
  case T:                              \
    SHUFFLE_ROWS(CUDA_T);              \
    break;


template <typename scalar_t>
__global__ void apply_shuffle_mul_sum_kernel(
    const scalar_t* __restrict__ input_tensor,  
    scalar_t* __restrict__ output_tensor,       
    const int32_t* __restrict__ permutation,    
    int m,
    int topk,
    int row_stride,
    const scalar_t* __restrict__ factors)  
{
  int i = blockIdx.x;
  if (i >= m) {
    return;
  }

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using t = float;
  using vec_t = flashinfer::vec_t<t, vec_size>;
  int thread_idx = threadIdx.x;
  int stride = blockDim.x;

  for (int d_vec_idx = thread_idx; d_vec_idx < row_stride / vec_size; d_vec_idx += stride) {
    int d = d_vec_idx * vec_size;
    vec_t sum_vec;
    sum_vec.fill(0.0f);

    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];

      vec_t val_vec;
      val_vec.cast_load(input_tensor + src_row * row_stride + d);

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }

#pragma unroll
      for (int k = 0; k < vec_size; ++k) {
        sum_vec[k] += factor * val_vec[k];
      }
    }
    sum_vec.cast_store(output_tensor + i * row_stride + d);
  }

  
  int remainder_start = (row_stride / vec_size) * vec_size;
  for (int d = remainder_start + thread_idx; d < row_stride; d += stride) {
    t sum_val = 0.0;
    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];
      t val = input_tensor[src_row * row_stride + d];

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }
      sum_val += factor * val;
    }
    output_tensor[i * row_stride + d] = sum_val;
  }
}



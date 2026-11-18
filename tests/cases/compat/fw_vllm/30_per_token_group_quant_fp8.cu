#include <cuda_runtime.h>

#pragma once


#include <cmath>

#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#pragma once


namespace vllm {


template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, __nv_fp8_e4m3> ||
                std::is_same_v<quant_type_t, __nv_fp8_e4m3>);
  quant_type_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

}

#pragma once

namespace vllm {

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst, const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op,       
    ScaOp&& scalar_op) {  
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);       
  constexpr int OUT_WIDTH = VEC_SIZE * sizeof(OutT);  
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);
  uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

  
  
  
  
  
  
  
  bool can_vec = ((addr & (WIDTH - 1)) == 0) &&
                 ((out_addr & (OUT_WIDTH - 1)) == 0) &&
                 ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      
      vin_t src = v_in[i];  
      vec_op(tmp, src);
      v_out[i] = tmp;  
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);       
  int alignment_bytes = WIDTH - misalignment_offset;  
  int prefix_elems = alignment_bytes & (WIDTH - 1);   
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);  

  
  
  
  if (((out_addr + prefix_elems * sizeof(OutT)) & (OUT_WIDTH - 1)) != 0) {
    for (int i = tid; i < len; i += stride) {
      scalar_op(out[i], in[i]);
    }
    return;
  }

  
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  
  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    
    vin_t src = v_in[i];  
    vec_op(tmp, src);
    v_out[i] = tmp;  
  }

  
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT* in,
                                                         OutT* out, int len,
                                                         int tid, int stride,
                                                         ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}

template <int VEC_SIZE, typename InT, typename ScaOp>
struct DefaultReadVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(src.val[i]);
    }
  }
};


template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_with_alignment(const InT* in, int len,
                                                     int tid, int stride,
                                                     VecOp&& vec_op,
                                                     ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);

    for (int i = tid; i < num_vec; i += stride) {
      vin_t tmp = v_in[i];
      vec_op(tmp);
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(in[i]);
  }

  in += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);

  
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }

  
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(in[i]);
  }
}


template <int VEC_SIZE, typename InT, typename ScaOp>
__device__ __forceinline__ void vectorize_read_with_alignment(
    const InT* in, int len, int tid, int stride, ScaOp&& scalar_op) {
  using Vec = DefaultReadVecOp<VEC_SIZE, InT, std::decay_t<ScaOp>>;
  vectorize_read_with_alignment<VEC_SIZE>(in, len, tid, stride, Vec{scalar_op},
                                          std::forward<ScaOp>(scalar_op));
}

}


#pragma once


#ifdef USE_ROCM

#else

#endif


#define VLLM_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
  switch (VEC_SIZE) {                         \
    case 16: {                                \
      constexpr int vec_size = 16;            \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 8: {                                 \
      constexpr int vec_size = 8;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 4: {                                 \
      constexpr int vec_size = 4;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    case 2: {                                 \
      constexpr int vec_size = 2;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
    default: {                                \
      constexpr int vec_size = 1;             \
      __VA_ARGS__();                          \
      break;                                  \
    }                                         \
  }

#define VLLM_DISPATCH_BOOL(expr, const_expr, ...) \
  if (expr) {                                     \
    constexpr bool const_expr = true;             \
    __VA_ARGS__();                                \
  } else {                                        \
    constexpr bool const_expr = false;            \
    __VA_ARGS__();                                \
  }

#define VLLM_DISPATCH_GROUP_SIZE(group_size, const_group_size, ...) \
  if (group_size == 128) {                                          \
    constexpr int const_group_size = 128;                           \
    __VA_ARGS__();                                                  \
  } else if (group_size == 64) {                                    \
    constexpr int const_group_size = 64;                            \
    __VA_ARGS__();                                                  \
  }


#pragma once


#ifdef TORCH_TARGET_VERSION
#else
#endif


__device__ __forceinline__ float GroupReduceMax(float val) {
#ifdef USE_ROCM
  
  
  const int lane_in_wave = threadIdx.x % warpSize;
  const unsigned long long mask = 0xFFFFull << ((lane_in_wave / 16) * 16);
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1, 16));
#else
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
#endif
  return val;
}

template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float ComputeGroupScale(
    const T* __restrict__ group_input, T* __restrict__ smem_group,
    const int group_size, const int lane_id, const int threads_per_group,
    const float eps, const float max_8bit) {
  float local_absmax = eps;

  constexpr int vec_size = 16 / sizeof(T);

  
  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(
      group_input,        
      smem_group,         
      group_size,         
      lane_id,            
      threads_per_group,  
      scalar_op_cache);   

  local_absmax = GroupReduceMax(local_absmax);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  return y_s;
}

template <typename T, typename DST_DTYPE>
__device__ __forceinline__ void QuantizeGroup(
    const T* __restrict__ smem_group, DST_DTYPE* __restrict__ group_output,
    const int group_size, const int lane_id, const int threads_per_group,
    const float y_s, const float min_8bit, const float max_8bit) {
  constexpr int vec_size = 16 / sizeof(T);

  
  auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = DST_DTYPE(q);
  };

  vllm::vectorize_with_alignment<vec_size>(
      smem_group,         
      group_output,       
      group_size,         
      lane_id,            
      threads_per_group,  
      scalar_op_quant);   
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s, const int group_size,
    const int num_groups, const int groups_per_block, const float eps,
    const float min_8bit, const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  using scale_element_t = float;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  const float y_s = ComputeGroupScale<T, SCALE_UE8M0>(
      group_input, smem_group, group_size, lane_id, threads_per_group, eps,
      max_8bit);

  scale_element_t y_s_quant = y_s;

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  __syncthreads();

  QuantizeGroup<T, DST_DTYPE>(smem_group, group_output, group_size, lane_id,
                              threads_per_group, y_s, min_8bit, max_8bit);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}


inline int GetGroupsPerBlockX(int64_t padded_groups_per_row) {
  if (padded_groups_per_row % 16 == 0) {
    return 16;
  }
  if (padded_groups_per_row % 8 == 0) {
    return 8;
  }
  return 4;
}


template <typename T, typename DST_DTYPE, int GROUP_SIZE, int kGroupsPerBlockX,
          int kRowsPerBlock>
__global__ void per_token_group_quant_8bit_packed_register_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    unsigned int* __restrict__ output_s_packed, const int padded_groups_per_row,
    const int groups_per_row, const int mn, const int output_q_mn_extent,
    const int tma_aligned_mn, const int64_t num_scale_elems, const float eps,
    const float min_8bit, const float max_8bit) {
  static_assert(GROUP_SIZE == 128, "fast path supports GROUP_SIZE==128");
  constexpr int THREADS_PER_GROUP = 8;
  constexpr int VEC_SIZE = 32 / sizeof(T);  
  static_assert(GROUP_SIZE == THREADS_PER_GROUP * VEC_SIZE,
                "GROUP_SIZE must equal THREADS_PER_GROUP * VEC_SIZE");
  static_assert(32 % THREADS_PER_GROUP == 0,
                "THREADS_PER_GROUP must divide warp size for the shuffle "
                "mask to be valid");
  static_assert(
      kGroupsPerBlockX > 0 && (kGroupsPerBlockX & (kGroupsPerBlockX - 1)) == 0,
      "kGroupsPerBlockX must be a positive power of 2");
  static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;

  const int sf_k_local = local_group_id % kGroupsPerBlockX;
  const int row_local = local_group_id / kGroupsPerBlockX;
  
  const int sf_k_idx = blockIdx.y * kGroupsPerBlockX + sf_k_local;
  const int mn_idx = blockIdx.x * kRowsPerBlock + row_local;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (mn_idx >= tma_aligned_mn) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  const bool is_valid_group = (mn_idx < mn) && (sf_k_idx < groups_per_row);

  
  
  
  
  
  
  alignas(16) T regs[VEC_SIZE];
  float local_absmax = eps;
  if (is_valid_group) {
    const T* group_input =
        input + static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
        sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
    const uint4* src = reinterpret_cast<const uint4*>(group_input);
    dst[0] = src[0];
    dst[1] = src[1];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = fabsf(static_cast<float>(regs[i]));
      local_absmax = fmaxf(local_absmax, v);
    }
  }

  
  
#ifdef USE_ROCM
  const int lane_in_wave = threadIdx.x % warpSize;
  const unsigned long long mask = 0xFFull << (lane_in_wave & ~7);
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 4, 8));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 2, 8));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 1, 8));
#else
  unsigned mask = 0xffu << (threadIdx.x & 24u);
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 4));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 2));
  local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, 1));
#endif

  float y_s = local_absmax / max_8bit;
  y_s = fmaxf(y_s, 1e-10f);
  uint32_t bits = __float_as_uint(y_s);
  uint32_t exp_bits = (bits >> 23) & 0xffu;
  uint32_t mant_bits = bits & 0x7fffffu;
  uint8_t exp_byte =
      static_cast<uint8_t>(exp_bits + (mant_bits != 0u ? 1u : 0u));

  
  if (lane_id == 0) {
    const int sf_k_pack_idx = sf_k_idx / 4;
    const int pos = sf_k_idx % 4;
    const int out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx;
    if (is_valid_group) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = exp_byte;
    } else if (out_idx < num_scale_elems) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = 0;
    }
  }

  
  
  
  if (!is_valid_group) {
    if (sf_k_idx < groups_per_row && mn_idx >= mn &&
        mn_idx < output_q_mn_extent) {
      DST_DTYPE* group_output =
          static_cast<DST_DTYPE*>(output_q) +
          static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
          sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
      *reinterpret_cast<uint4*>(group_output) = make_uint4(0, 0, 0, 0);
    }
    return;
  }

  
  float y_s_q = __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);
  float inv_y = 1.0f / y_s_q;

  
  
  uint32_t packed_lo = 0;
  uint32_t packed_lo_hi = 0;
  uint32_t packed_hi_lo = 0;
  uint32_t packed_hi = 0;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float q =
        fminf(fmaxf(static_cast<float>(regs[i]) * inv_y, min_8bit), max_8bit);
    DST_DTYPE qb = DST_DTYPE(q);
    uint8_t byte = *reinterpret_cast<uint8_t*>(&qb);
    const int shift = (i & 3) * 8;
    if (i < 4) {
      packed_lo |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 8) {
      packed_lo_hi |= static_cast<uint32_t>(byte) << shift;
    } else if (i < 12) {
      packed_hi_lo |= static_cast<uint32_t>(byte) << shift;
    } else {
      packed_hi |= static_cast<uint32_t>(byte) << shift;
    }
  }

  uint4 packed_out =
      make_uint4(packed_lo, packed_lo_hi, packed_hi_lo, packed_hi);
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) +
      static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
      sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
  *reinterpret_cast<uint4*>(group_output) = packed_out;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}




#include <cmath>


#pragma once


#define VLLM_STABLE_DISPATCH_FP8_CASE(enum_type, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)


#define VLLM_STABLE_DISPATCH_IDX_CASE(enum_type, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, idx_t, __VA_ARGS__)


#define VLLM_STABLE_DISPATCH_IDX_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                       \
                      VLLM_STABLE_DISPATCH_CASE_IDX_TYPES(__VA_ARGS__))


#define VLLM_STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                            \
                      VLLM_STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))


#define VLLM_STABLE_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                            \
                      VLLM_STABLE_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define VLLM_STABLE_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(                                                    \
      TYPE, NAME,                                                         \
      VLLM_STABLE_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(__VA_ARGS__))


#ifdef USE_ROCM
#else
#endif


#define VLLM_STABLE_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                       \
                      VLLM_STABLE_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))


#define VLLM_STABLE_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                        \
                      VLLM_STABLE_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))


#ifdef USE_ROCM
#else
#endif

#define VLLM_STABLE_DISPATCH_QUANT_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                         \
                      VLLM_STABLE_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__))


#define VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, const_group_size, ...) \
  if (group_size == 128) {                                                 \
    constexpr int const_group_size = 128;                                  \
    __VA_ARGS__();                                                         \
  } else if (group_size == 64) {                                           \
    constexpr int const_group_size = 64;                                   \
    __VA_ARGS__();                                                         \
  }


#define VLLM_STABLE_DISPATCH_BOOL(expr, const_expr, ...) \
  if (expr) {                                            \
    constexpr bool const_expr = true;                    \
    __VA_ARGS__();                                       \
  } else {                                               \
    constexpr bool const_expr = false;                   \
    __VA_ARGS__();                                       \
  }


#define VLLM_STABLE_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
  switch (VEC_SIZE) {                                \
    case 16: {                                       \
      constexpr int vec_size = 16;                   \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 8: {                                        \
      constexpr int vec_size = 8;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 4: {                                        \
      constexpr int vec_size = 4;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 2: {                                        \
      constexpr int vec_size = 2;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    default: {                                       \
      constexpr int vec_size = 1;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
  }


#pragma once


#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>


inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;


#pragma once
#pragma once


namespace vllm {


template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};


template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;

}


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

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #if CUB_VERSION >= 200800
    #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
  #else   
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
  #endif  
#else
  #include <hipcub/hipcub.hpp>
namespace cub = hipcub;
using CubAddOp = hipcub::Sum;
using CubMaxOp = hipcub::Max;
#endif  


static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  
  
  
  
  float dst = std::nearbyint(x);

  
  
  
  
  
  
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  
  
  
  
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);

  
  
  
  
  float dst = std::nearbyint(x);

  
  if (dst >= i32_max_f) {
    return i32_max;
  }
  
  if (dst <= i32_min_f) {
    return i32_min;
  }

  return static_cast<int32_t>(dst);
#else
  
  uint32_t dst;
  asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int32_t&>(dst);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  
  
  
  
  
  
  int32_t dst = (x < i8_min) ? i8_min : (x > i8_max) ? i8_max : x;
  return static_cast<int8_t>(dst);
#else
  
  uint32_t dst;
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_t>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;
  const float scale = *scale_ptr;

  
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(static_cast<float>(src) / scale);
      });
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void static_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const azp_t* azp_ptr, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;
  const float scale = *scale_ptr;
  const azp_t azp = *azp_ptr;
  const float inv_s = 1.0f / scale;

  
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        const auto v = static_cast<float>(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

template <typename scalar_t, typename scale_t>
__global__ void dynamic_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  
  float thread_max = 0.f;
  vectorize_read_with_alignment<16>(
      row_in, hidden_size, tid, stride, [&] __device__(const scalar_t& src) {
        const float v = fabsf(static_cast<float>(src));
        thread_max = fmaxf(thread_max, v);
      });
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);
  __shared__ float absmax;
  if (tid == 0) {
    absmax = block_max;
    scale_out[blockIdx.x] = absmax / 127.f;
  }
  __syncthreads();

  float inv_s = (absmax == 0.f) ? 0.f : 127.f / absmax;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(static_cast<float>(src) * inv_s);
      });
}


struct MinMax {
  float min, max;

  __host__ __device__ MinMax()
      : min(std::numeric_limits<float>::max()),
        max(std::numeric_limits<float>::lowest()) {}

  __host__ __device__ explicit MinMax(float v) : min(v), max(v) {}

  __host__ __device__ MinMax& operator+=(float v) {
    min = fminf(min, v);
    max = fmaxf(max, v);
    return *this;
  }

  
  __host__ __device__ MinMax& operator&=(const MinMax& other) {
    min = fminf(min, other.min);
    max = fmaxf(max, other.max);
    return *this;
  }
};

__host__ __device__ inline MinMax operator+(MinMax a, float v) {
  return a += v;
}
__host__ __device__ inline MinMax operator&(MinMax a, const MinMax& b) {
  return a &= b;
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, azp_t* azp_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  MinMax thread_mm;
  vectorize_read_with_alignment<16>(row_in, hidden_size, tid, stride,
                                    [&] __device__(const scalar_t& src) {
                                      thread_mm += static_cast<float>(src);
                                    });

  using BlockReduce = cub::BlockReduce<MinMax, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;

  MinMax mm = BlockReduce(tmp).Reduce(
      thread_mm,
      [] __device__(MinMax a, const MinMax& b) {
        a &= b;
        return a;
      },
      blockDim.x);

  __shared__ float scale_sh;
  __shared__ azp_t azp_sh;
  if (tid == 0) {
    float s = (mm.max - mm.min) / 255.f;
    float zp = nearbyintf(-128.f - mm.min / s);  
    scale_sh = s;
    azp_sh = azp_t(zp);
    scale_out[blockIdx.x] = s;
    azp_out[blockIdx.x] = azp_sh;
  }
  __syncthreads();

  const float inv_s = 1.f / scale_sh;
  const azp_t azp = azp_sh;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        const auto v = static_cast<float>(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

}


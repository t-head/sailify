


#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#endif


#include <cstdint>


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


#ifndef USE_ROCM
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
#else
using bf16_t = __hip_bfloat16;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif


static constexpr uint32_t kWarpSize = 32;


template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVec {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) {
    return data[i];
  }
  __device__ __forceinline__ T operator[](int i) const {
    return data[i];
  }
  __device__ __forceinline__ void load(const void* ptr, int64_t offset = 0) {
    *this = reinterpret_cast<const AlignedVec*>(ptr)[offset];
  }
  __device__ __forceinline__ void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<AlignedVec*>(ptr)[offset] = *this;
  }
};

__device__ __forceinline__ float bf16_to_float(bf16_t v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ bf16_t float_to_bf16(float v) {
#ifndef USE_ROCM
  return __float2bfloat16_rn(v);
#else
  return __float2bfloat16(v);
#endif
}


__device__ __forceinline__ int32_t cast_to_ue8m0(float x) {
  uint32_t u = __float_as_uint(x);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu);
  uint32_t mant = u & 0x7FFFFFu;
  return exp + (mant != 0);
}

__device__ __forceinline__ float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float(static_cast<uint32_t>((127 + 127 - exp) << 23));
}

static constexpr float kFP8Max = 448.0f;

#ifndef USE_ROCM
__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  x = fmaxf(fminf(x, kFP8Max), -kFP8Max);
  y = fmaxf(fminf(y, kFP8Max), -kFP8Max);
  return __nv_fp8x2_e4m3(float2{x, y});
}
#elif HIP_FP8_TYPE_OCP && !HIP_FP8_TYPE_FNUZ


__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  const float2 v{fmaxf(fminf(x, kFP8Max), -kFP8Max), fmaxf(fminf(y, kFP8Max), -kFP8Max)};
  return __hip_cvt_float2_to_fp8x2(v, __HIP_NOSAT, __HIP_E4M3);
}
#else

__device__ __forceinline__ uint8_t cvt_float_to_fp8_e4m3(float val) {
  constexpr float kMax = kFP8Max;
  val = fmaxf(fminf(val, kMax), -kMax);
  if (val == 0.0f) return 0;

  uint32_t f32 = __float_as_uint(val);
  uint8_t sign = static_cast<uint8_t>((f32 >> 24) & 0x80u);
  f32 &= 0x7FFFFFFFu;

  int32_t exp32 = static_cast<int32_t>((f32 >> 23) & 0xFFu);
  uint32_t mant32 = f32 & 0x7FFFFFu;

  
  int32_t exp8 = exp32 - 120;

  if (exp8 <= 0) {
    mant32 |= 0x800000u;
    int32_t shift = 1 - exp8;
    if (shift > 24) return sign;
    uint32_t shifted = mant32 >> (20 + shift);
    uint32_t rbit = (shift <= 23) ? ((mant32 >> (19 + shift)) & 1u) : 0u;
    uint32_t sbit = (shift <= 23) ? ((mant32 & ((1u << (19 + shift)) - 1u)) != 0) : 0u;
    shifted += (rbit && (sbit || (shifted & 1u)));
    return sign | static_cast<uint8_t>(shifted & 0x7u);
  }
  if (exp8 >= 15) return sign | 0x7Eu;

  uint32_t mant3 = (mant32 >> 20) & 0x7u;
  uint32_t rbit = (mant32 >> 19) & 1u;
  uint32_t sbit = (mant32 & 0x7FFFFu) != 0;
  mant3 += (rbit && (sbit || (mant3 & 1u)));
  if (mant3 > 7) {
    mant3 = 0;
    exp8++;
    if (exp8 >= 15) return sign | 0x7Eu;
  }
  return sign | (static_cast<uint8_t>(exp8) << 3) | static_cast<uint8_t>(mant3);
}

__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  uint8_t x8 = cvt_float_to_fp8_e4m3(x);
  uint8_t y8 = cvt_float_to_fp8_e4m3(y);
  return static_cast<uint16_t>(x8) | (static_cast<uint16_t>(y8) << 8);
}
#endif


namespace {

constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / kWarpSize;

constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / kWarpSize;

struct FusedQNormRopeParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_output;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  uint32_t batch_size;
  uint32_t num_q_heads;
  float eps;
};


template <int64_t kHeadDim, int64_t kRopeDim>
struct QKernelTraits {
  static constexpr int64_t kMaxVecSize = 16 / sizeof(bf16_t);  
  
  
  
  static constexpr int64_t kVecSize = kRopeDim / kWarpSize;  
  static constexpr int64_t kLocalSize = kHeadDim / (kWarpSize * kVecSize);
  static constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpSize * kVecSize) == 0);
  static_assert(kRopeDim % kVecSize == 0);
  static_assert(kRopeDim == kWarpSize * 2, "1 (real, imag) pair per lane");
};

template <int64_t kHeadDim, int64_t kRopeDim>
__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_norm_rope_kernel(
    const __grid_constant__ FusedQNormRopeParams params) {
  using Traits = QKernelTraits<kHeadDim, kRopeDim>;
  constexpr int64_t kVecSize = Traits::kVecSize;
  constexpr int64_t kLocalSize = Traits::kLocalSize;
  constexpr uint32_t kRopeSize = Traits::kRopeSize;

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float2 = AlignedVec<float, 2>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;

  const uint32_t total_works = params.batch_size * params.num_q_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_q_heads;
  const uint32_t head_id = work_id % params.num_q_heads;
  const auto input_ptr =
      static_cast<const bf16_t*>(params.q_input) + batch_id * params.q_input_stride_batch + head_id * kHeadDim;
  const auto output_ptr =
      static_cast<bf16_t*>(params.q_output) + batch_id * params.q_output_stride_batch + head_id * kHeadDim;
  const auto position = params.positions[batch_id];

  __shared__ Storage s_rope[kFusedQNumWarps][kRopeSize];

  
  Float2 freq;
  freq.load(params.freqs_cis + position * kRopeDim, lane_id);

  
  Storage input_vec[kLocalSize];
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    input_vec[i].load(input_ptr, lane_id + i * kWarpSize);
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float x = bf16_to_float(input_vec[i][j]);
      sum_of_squares += x * x;
    }
  }
  const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float x = bf16_to_float(input_vec[i][j]);
      input_vec[i][j] = float_to_bf16(x * norm_factor);
    }
  }

  
  const bool is_rope_lane = lane_id >= kWarpSize - kRopeSize;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    if (i == kLocalSize - 1 && is_rope_lane) {
      const auto rope_id = lane_id - (kWarpSize - kRopeSize);
      s_rope[warp_id][rope_id] = input_vec[i];
    } else {
      input_vec[i].store(output_ptr, lane_id + i * kWarpSize);
    }
  }
  __syncwarp();

  
  auto elem_ptr = reinterpret_cast<bf16x2_t*>(&s_rope[warp_id][0]);
  bf16x2_t elem = elem_ptr[lane_id];
#ifndef USE_ROCM
  float2 elem_f = __bfloat1622float2(elem);
  float x_real = elem_f.x, x_imag = elem_f.y;
#else
  float x_real = __bfloat162float(elem.x), x_imag = __bfloat162float(elem.y);
#endif
  float freq_real = freq[0], freq_imag = freq[1];
  float rot_real = x_real * freq_real - x_imag * freq_imag;
  float rot_imag = x_real * freq_imag + x_imag * freq_real;
  bf16x2_t rotated = __float22bfloat162_rn(make_float2(rot_real, rot_imag));
  auto out_elem = reinterpret_cast<bf16x2_t*>(output_ptr + (kHeadDim - kRopeDim));
  out_elem[lane_id] = rotated;
}


struct FusedKNormRopeFlashMLAParams {
  const void* __restrict__ kv;
  const void* __restrict__ kv_weight;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  const int32_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <int64_t kHeadDim, int64_t kRopeDim, int32_t kPageBits>
__global__ __launch_bounds__(kFusedKBlockSize, 8) void fused_k_norm_rope_flashmla_kernel(
    const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kPageBytes = ((584ll << kPageBits) + 575) / 576 * 576;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpSize;
  const auto lane_id = tx % kWarpSize;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const bf16_t*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = params.positions[work_id];
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  AlignedVec<float, kVecSize> data, freq;

  
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float x = bf16_to_float(input_vec[i]);
      sum_of_squares += x * x;
    }
    __syncthreads();
    const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float x = bf16_to_float(input_vec[i]);
      float w = bf16_to_float(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 576;

  
  if (warp_id == kRopeWarp) {
    float x_real = data[0], x_imag = data[1];
    float freq_real = freq[0], freq_imag = freq[1];
    float rot_real = x_real * freq_real - x_imag * freq_imag;
    float rot_imag = x_real * freq_imag + x_imag * freq_real;
    auto rope_ptr = value_ptr + 448;
  } else {
    float x = data[0], y = data[1];
    auto scale_ptr = page_ptr + (576ll << kPageBits) + offset * 8;
  }
}


struct FusedQIndexerRopeHadamardQuantParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_fp8;
  const void* __restrict__ weight;
  float* __restrict__ weights_out;
  float weight_scale;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  uint32_t batch_size;
  uint32_t num_heads;
};

__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_kernel(
    const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpSize - kRopeSize;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const bf16_t*>(params.q_input) + work_id * kHeadDim;
  const auto position = params.positions[batch_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  Float4 data, freq;
  const float weight_val = bf16_to_float(static_cast<const bf16_t*>(params.weight)[work_id]);

  
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpSize - kRopeSize));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] = bf16_to_float(input_vec[i]);
  }

  
  if (is_rope_lane) {
    float x_r = data[0], x_i = data[1], y_r = data[2], y_i = data[3];
    float fxr = freq[0], fxi = freq[1], fyr = freq[2], fyi = freq[3];
    data[0] = x_r * fxr - x_i * fxi;
    data[1] = x_r * fxi + x_i * fxr;
    data[2] = y_r * fyr - y_i * fyi;
    data[3] = y_r * fyi + y_i * fyr;
  }

  
  {
    {
      float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
      }
    }
    const float kHadamardScale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  
  {
    float local_max = fabsf(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i)
      local_max = fmaxf(local_max, fabsf(data[i]));


    auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
  }
}

}



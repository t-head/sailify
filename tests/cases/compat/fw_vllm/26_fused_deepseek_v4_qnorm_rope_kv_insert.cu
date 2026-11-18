


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


#include <cmath>
#pragma once

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#endif

#ifdef USE_ROCM
struct Utils {
  static __host__ int get_warp_size() {
    static bool is_cached = false;
    static int result;

    if (!is_cached) {
      int device_id;
      cudaDeviceProp deviceProp;
      cudaGetDevice(&device_id);
      cudaGetDeviceProperties(&deviceProp, device_id);

      result = deviceProp.warpSize;
      is_cached = true;
    }

    return result;
  }

  static __device__ constexpr int get_warp_size() {
  #ifdef __GFX9__
    return 64;
  #else
    return 32;
  #endif
  }
};

  #define WARP_SIZE Utils::get_warp_size()
#else
  #define WARP_SIZE 32
#endif

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif


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


#ifndef USE_ROCM
  #include <cuda.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace vllm {


template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

template <>
struct _typeConvert<float> {
  static constexpr bool exists = true;
  using hip_type = float;
  using packed_hip_type = float2;
  using packed_hip_type4 = float4;  

  __device__ static __forceinline__ float convert(hip_type x) { return x; }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return x;
  }
  __device__ static __forceinline__ float4 convert(packed_hip_type4 x) {
    return x;
  }
};

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))


  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(USE_ROCM)


  #endif  
          
#endif    
          


template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  

  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] += other.data[i];
          data[i + 1] += other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp += T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] *= other.data[i];
          data[i + 1] *= other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp *= T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};
}


#ifndef USE_ROCM
  #include <cuda_fp8.h>
#else
  #include <hip/hip_fp8.h>
#endif
#include <cuda_runtime.h>
#include <type_traits>

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

#ifdef USE_ROCM

__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  
  #if defined(__gfx942__)
  __hip_fp8_e4m3_fnuz fp8_val(val);
  #else
  __hip_fp8_e4m3 fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
#endif

namespace vllm {
namespace deepseek_v4_fused_ops {

namespace {
inline int getSMVersion() {
}
}


constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;   
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;  


#if defined(USE_ROCM) && defined(__gfx942__)
constexpr float kFp8Max = 224.0f;
#else
constexpr float kFp8Max = 448.0f;
#endif

#ifndef USE_ROCM


constexpr float NUM_TOKEN_CUTOFF = 1024;
#endif


constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  


__device__ __forceinline__ uint4 packFp8E4M3x16(float const* values,
                                                float const scale) {
#ifndef USE_ROCM
  uint4 out;
  auto* out2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 scaled =
        make_float2(values[2 * i] * scale, values[2 * i + 1] * scale);
    scaled.x = fminf(fmaxf(scaled.x, -kFp8Max), kFp8Max);
    scaled.y = fminf(fmaxf(scaled.y, -kFp8Max), kFp8Max);
    out2[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
  }
  return out;
#else
  uint8_t out_bytes[kElemsPerLane];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    float scaled = values[i] * scale;
    scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
    out_bytes[i] = rocm_cvt_float_to_fp8_e4m3(scaled);
  }
  return *reinterpret_cast<uint4 const*>(out_bytes);
#endif
}


__device__ __forceinline__ float warp4MaxAbs(float val) {
  
  float peer = __shfl_xor_sync(FINAL_MASK, val, 1);
  val = fmaxf(val, peer);
  peer = __shfl_xor_sync(FINAL_MASK, val, 2);
  val = fmaxf(val, peer);
  return val;
}

template <typename T>
__device__ __forceinline__ float warpSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}


template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel(
    scalar_t_in const* __restrict__ q_in,      
    scalar_t_in* __restrict__ q_out,           
    scalar_t_in const* __restrict__ kv_in,     
    uint8_t* __restrict__ k_cache,             
    int64_t const* __restrict__ slot_mapping,  
    int64_t const* __restrict__ position_ids,  
    float const* __restrict__ cos_sin_cache,   
    float const eps,
    int const num_tokens_full,    
    int const num_tokens_insert,  
    int const num_heads_q,        
    int const cache_block_size,   
    int const kv_block_stride) {  
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  
  
  
  if constexpr (std::is_same_v<scalar_t_in, __nv_bfloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    constexpr int kTotalSlotsPerToken = kNumHeadsQPadded + 1;
    int const tokenIdx = globalWarpIdx / kTotalSlotsPerToken;
    int const slotIdx = globalWarpIdx % kTotalSlotsPerToken;
    if (tokenIdx >= num_tokens_full) return;

    bool const isKV = (slotIdx == kNumHeadsQPadded);
    bool const isPadQ = !isKV && (slotIdx >= num_heads_q);
    
    if (isKV && tokenIdx >= num_tokens_insert) return;

    
    
    
    
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    
    int const dim_base = laneId * kElemsPerLane;  

    
    
    uint4 v0, v1;
    if (!isPadQ) {
      scalar_t_in const* src_ptr;
      if (isKV) {
        src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        int64_t const q_row_offset =
            (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) *
                kHeadDim +
            dim_base;
        src_ptr = q_in + q_row_offset;
      }
      v0 = *reinterpret_cast<uint4 const*>(src_ptr);
      v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);
    }

    processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
        v0, v1, tokenIdx, slotIdx, dim_base, laneId, num_heads_q, eps, q_out,
        k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size,
        kv_block_stride);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}


template <typename scalar_t_in, int kNumHeadsQPadded>
__global__ void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid(
    scalar_t_in const* __restrict__ q_in, scalar_t_in* __restrict__ q_out,
    scalar_t_in const* __restrict__ kv_in, uint8_t* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, float const eps,
    int const num_tokens_full, int const num_tokens_insert,
    int const num_heads_q, int const cache_block_size,
    int const kv_block_stride) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, __nv_bfloat16>) {
    return;
  } else {
#endif
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const tokenIdx = blockIdx.x;
    if (tokenIdx >= num_tokens_full) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const dim_base = laneId * kElemsPerLane;  
    
    int const slot_end = (tokenIdx >= num_tokens_insert)
                             ? kNumHeadsQPadded
                             : (kNumHeadsQPadded + 1);

    auto load_slot = [&](int s, uint4& va, uint4& vb) {
      
      if (s >= num_heads_q && s < kNumHeadsQPadded) return;
      scalar_t_in const* src;
      if (s == kNumHeadsQPadded) {
        src = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
      } else {
        src = q_in +
              (static_cast<int64_t>(tokenIdx) * num_heads_q +
               static_cast<int64_t>(s)) *
                  kHeadDim +
              dim_base;
      }
      va = *reinterpret_cast<uint4 const*>(src);
      vb = *reinterpret_cast<uint4 const*>(src + 8);
    };

    if (warpId < slot_end) {
      int curr_slot = warpId;
      uint4 v0_curr, v1_curr;
      load_slot(curr_slot, v0_curr, v1_curr);

      while (curr_slot < slot_end) {
        int const next_slot = curr_slot + warpsPerBlock;
        bool const has_next = (next_slot < slot_end);

        
        uint4 v0_next, v1_next;
        if (has_next) {
          load_slot(next_slot, v0_next, v1_next);
        }

        processDeepseekV4Slot<scalar_t_in, kNumHeadsQPadded>(
            v0_curr, v1_curr, tokenIdx, curr_slot, dim_base, laneId,
            num_heads_q, eps, q_out, k_cache, slot_mapping, position_ids,
            cos_sin_cache, cache_block_size, kv_block_stride);

        
        v0_curr = v0_next;
        v1_curr = v1_next;
        curr_slot = next_slot;
      }  
    }  

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}


}
}






#include <type_traits>

#include <cuda_runtime.h>

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

#pragma once


#ifdef TORCH_TARGET_VERSION
#else
#endif

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

template <typename HashIndType>
__device__ __forceinline__ int64_t load_index_as_int64(const HashIndType* ptr,
                                                       int64_t offset) {
  return static_cast<int64_t>(ptr[offset]);
}


template <typename T,
          
          int N,
          
          int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  }
}

#ifndef USE_ROCM


template <typename OutIndType, typename HashIndType>
__launch_bounds__(128) __global__
    void dsv4HashTopkSoftplusSqrt(const float* input, float* output,
                                  OutIndType* indices, int num_rows,
                                  int num_experts, float routed_scaling_factor,
                                  const HashIndType* input_ids,
                                  const HashIndType* tid2eid,
                                  const bool* is_padding) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= num_rows) return;
  const int64_t token_id = load_index_as_int64(input_ids, warp);
  const bool is_pad_row = is_padding != nullptr && is_padding[warp];

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  #endif
  int expert = 0;
  float weight = 0.f;
  if (lane < 6 && !is_pad_row) {
    
    expert = static_cast<int>(tid2eid[token_id * 6 + lane]);
    const float x = input[warp * num_experts + expert];
    weight = sqrtf(fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x))));
    if (isnan(weight)) {
      weight = 0.f;
    }
  }
  float weight_sum = weight;
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    
    weight_sum += VLLM_SHFL_XOR_SYNC(weight_sum, mask);
  }

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
  #endif
  if (lane < 6) {
    const int offset = warp * 6 + lane;
    output[offset] =
        weight * routed_scaling_factor / (weight_sum > 0.f ? weight_sum : 1.f);
    indices[offset] = !is_pad_row ? static_cast<OutIndType>(expert)
                                  : static_cast<OutIndType>(-1);
  }
}

template <typename OutIndType, typename HashIndType>
void launchDsv4HashTopk(const float* input, float* output, OutIndType* indices,
                        int num_rows, int num_experts,
                        double routed_scaling_factor,
                        const HashIndType* input_ids,
                        const HashIndType* tid2eid, cudaStream_t stream,
                        const bool* is_padding) {
  if (num_rows == 0) return;
  auto* kernel = &dsv4HashTopkSoftplusSqrt<OutIndType, HashIndType>;
  cudaLaunchConfig_t config = {};
  config.gridDim = (num_rows + 3) / 4;
  config.blockDim = 128;
  config.stream = stream;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const float scale = static_cast<float>(routed_scaling_factor);
  cudaLaunchKernelEx(&config, kernel, input, output, indices, num_rows,
                     num_experts, scale, input_ids, tid2eid, is_padding);
}
#endif


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, bool USE_HASH, typename IndType,
          typename HashIndType, typename InputType = float>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGatingSoftplusSqrt(
        const InputType* input, const bool* finished, float* output,
        const int num_rows, IndType* indices, int* source_rows, const int k,
        const int start_expert, const int end_expert, const bool renormalize,
        double routed_scaling_factor, const float* correction_bias,
        const HashIndType* input_ids, const HashIndType* tid2eid,
        const bool* is_padding) {
  static_assert(std::is_same_v<InputType, float> ||
                    std::is_same_v<InputType, __nv_bfloat16> ||
                    std::is_same_v<InputType, __half>,
                "InputType must be float, __nv_bfloat16, or __half");

  
  
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  if constexpr (std::is_same_v<InputType, __nv_bfloat16> ||
                std::is_same_v<InputType, __half>) {
    static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
                  "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
  }

  
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM,
                "THREADS_PER_ROW can be at most warp size");

  
  static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  
  

  
  
  
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  
  
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  
  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;
  const bool is_pad_row = is_padding != nullptr && is_padding[thread_row];

  
  
  const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  
  
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  
  float row_chunk[VPT];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (is_pad_row) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = 0.f;
    }
  } else if constexpr (std::is_same_v<InputType, float>) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
    const VecType* vec_thread_read_ptr =
        reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
  } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr =
          reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2));
        }
      }
    } else {  
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __nv_bfloat16* scalar_ptr =
            thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __bfloat162float(*scalar_ptr);
      }
    }
  } else if constexpr (std::is_same_v<InputType, __half>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__half, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr =
          reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __half22float2(
              *reinterpret_cast<const __half2*>(vec.data + jj * 2));
        }
      }
    } else {  
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __half2float(*scalar_ptr);
      }
    }
  }
  constexpr float threshold = 20.0f;
  constexpr float beta = 1.0f;

  
  if constexpr (USE_HASH) {
    const int64_t token_id = load_index_as_int64(input_ids, thread_row);
    const int64_t token_expert_offset = token_id * static_cast<int64_t>(k);
    if (!is_pad_row) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        float val = row_chunk[ii];
        float val_b = val * beta;
        val = (val_b > threshold) ? val : (__logf(1.0f + __expf(val_b))) / beta;
        val = sqrtf(val);

        
        
        
        if (isnan(val)) {
          val = 0.f;
        }
        row_chunk[ii] = val;
      }
    }
    float selected_sum = 0.f;
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expert = static_cast<int>(
          load_index_as_int64(tid2eid, token_expert_offset + k_idx));
      const int idx = k * thread_row + k_idx;
      for (int ii = 0; ii < VPT; ++ii) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread +
                               group_id * THREADS_PER_ROW * ELTS_PER_LDG +
                               local_id;
        if (expert == expert_idx) {
          indices[idx] = !is_pad_row ? static_cast<IndType>(expert)
                                     : static_cast<IndType>(-1);
          selected_sum += row_chunk[ii];
          break;
        }
      }
    }
    
    
    
    if (renormalize) {
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        selected_sum +=
            VLLM_SHFL_XOR_SYNC_WIDTH(selected_sum, mask, THREADS_PER_ROW);
      }
    }
    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      scale /= denom;
    }

#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expert = static_cast<int>(
          load_index_as_int64(tid2eid, token_expert_offset + k_idx));
      const int idx = k * thread_row + k_idx;
      for (int ii = 0; ii < VPT; ++ii) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread +
                               group_id * THREADS_PER_ROW * ELTS_PER_LDG +
                               local_id;
        if (expert == expert_idx) {
          output[idx] = row_chunk[ii] * scale;
          break;
        }
      }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  } else {
    if (!is_pad_row) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        float val = row_chunk[ii];
        float val_b = val * beta;
        
        
        val = (val_b > threshold) ? val : (__logf(1.0f + __expf(val_b))) / beta;
        val = sqrtf(val);
        
        
        
        if (isnan(val)) {
          val = 0.f;
        }
        if (correction_bias) {
          const int group_id = ii / ELTS_PER_LDG;
          const int local_id = ii % ELTS_PER_LDG;
          const int expert_idx = first_elt_read_by_thread +
                                 group_id * THREADS_PER_ROW * ELTS_PER_LDG +
                                 local_id;
          val = val + correction_bias[expert_idx];
        }
        row_chunk[ii] = val;
      }
    }

    
    
    
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      
      float max_val = row_chunk[0];
      int expert = start_col;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];

          
          
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }


#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max =
            VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
        int other_expert =
            VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

        
        
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      
      if (thread_group_idx == 0) {
        
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row =
            row_is_active && node_uses_expert && !is_pad_row;

        
        
        
        const int idx = k * thread_row + k_idx;
        if (correction_bias != nullptr && should_process_row) {
          max_val -= correction_bias[expert];
        }
        output[idx] = max_val;
        indices[idx] =
            !is_pad_row ? expert - start_expert : static_cast<IndType>(-1);
        source_rows[idx] = k_idx * num_rows + thread_row;
        if (renormalize) {
          selected_sum += max_val;
        }
      }

      
      
      if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group =
            (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

        
        
        if (thread_group_idx == thread_to_clear_in_group) {
          const int offset_for_expert = expert % ELTS_PER_LDG;
          
          
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
              -10000.f;
        }
      }
    }

    
    if (thread_group_idx == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] = output[idx] * scale;
      }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
  }
}

namespace detail {


template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM,
          typename InputType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}

#define DISPATCH_HASH(use_hash, USE_HASH, ...)                                 \
  if (use_hash) {                                                              \
    const bool USE_HASH = true;                                                \
    static_assert(USE_HASH == true, "USE_HASH must be compile-time constant"); \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    const bool USE_HASH = false;                                               \
    static_assert(USE_HASH == false,                                           \
                  "USE_HASH must be compile-time constant");                   \
    __VA_ARGS__                                                                \
  }

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM,
          int MAX_BYTES_PER_LDG, typename IndType, typename HashIndType,
          typename InputType>
void topkGatingSoftplusSqrtLauncherHelper(
    const InputType* input, const bool* finished, float* output,
    IndType* indices, int* source_row, const int num_rows, const int k,
    const int start_expert, const int end_expert, const bool renormalize,
    double routed_scaling_factor, const float* correction_bias,
    const bool use_hash, const HashIndType* input_ids,
    const HashIndType* tid2eid, cudaStream_t stream, const bool* is_padding) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants =
      detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
  DISPATCH_HASH(use_hash, USE_HASH, {
    auto* kernel =
        &topkGatingSoftplusSqrt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,
                                WARP_SIZE_PARAM, USE_HASH, IndType, HashIndType,
                                InputType>;
#ifndef USE_ROCM
    cudaLaunchConfig_t config = {};
    config.gridDim = num_blocks;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, finished, output, num_rows,
                       indices, source_row, k, start_expert, end_expert,
                       renormalize, routed_scaling_factor, correction_bias,
                       input_ids, tid2eid, is_padding);
#else
    kernel<<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert,
        end_expert, renormalize, routed_scaling_factor, correction_bias,
        input_ids, tid2eid, is_padding);
#endif
  })
}

#ifndef USE_ROCM
  #define LAUNCH_SOFTPLUS_SQRT(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)           \
    static_assert(WARP_SIZE == 32,                                             \
                  "Unsupported warp size. Only 32 is supported for CUDA");     \
    topkGatingSoftplusSqrtLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, \
                                         MAX_BYTES>(                           \
        gating_output, nullptr, topk_weights, topk_indices,                    \
        token_expert_indices, num_tokens, topk, 0, num_experts, renormalize,   \
        routed_scaling_factor, correction_bias, use_hash, input_ids, tid2eid,  \
        stream, is_padding);
#else
  #define LAUNCH_SOFTPLUS_SQRT(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)           \
    if (WARP_SIZE == 64) {                                                     \
      topkGatingSoftplusSqrtLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 64,      \
                                           MAX_BYTES>(                         \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          routed_scaling_factor, correction_bias, use_hash, input_ids,         \
          tid2eid, stream, is_padding);                                        \
    } else if (WARP_SIZE == 32) {                                              \
      topkGatingSoftplusSqrtLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32,      \
                                           MAX_BYTES>(                         \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          routed_scaling_factor, correction_bias, use_hash, input_ids,         \
          tid2eid, stream, is_padding);                                        \
    } else {                                                                   \
      assert(false &&                                                          \
             "Unsupported warp size. Only 32 and 64 are supported for ROCm");  \
    }
#endif


}
}



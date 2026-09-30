


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


template <
    typename T,
    
    int N,
    
    int Alignment = sizeof(T) * N
>
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


enum ScoringFunc {
  SCORING_SOFTMAX = 0, 
  SCORING_SIGMOID = 1  
};


template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    float threadData(-FLT_MAX);

    
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData = max(val, threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData += expf(val - float_max);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        float softmax_val = expf(val - float_max) * normalizing_factor;
        
        if (isnan(softmax_val) || isinf(softmax_val)) softmax_val = 0.f;
        output[idx] = softmax_val;
    }
}

template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSigmoid(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    const int thread_row_offset = blockIdx.x * num_cols;

    
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        float sigmoid_val = 1.0f / (1.0f + __expf(-val));
        
        if (isnan(sigmoid_val) || isinf(sigmoid_val)) sigmoid_val = 0.f;
        output[idx] = sigmoid_val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* bias,
    const double routed_scaling_factor,
    const bool* is_padding)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; 

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;

            
            if (bias != nullptr) {
              inp_kvp.value = inputs_after_softmax[idx] + bias[expert];
            } else {
              inp_kvp.value = inputs_after_softmax[idx];
            }

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const bool is_pad_row = is_padding != nullptr && is_padding[block_row];

            const int idx = k * block_row + k_idx;
            
            output[idx] = inputs_after_softmax[thread_read_offset + expert];
            indices[idx] = is_pad_row ? static_cast<IndType>(-1)
                                       : (should_process_row ? (expert - start_expert) : num_experts);
            assert(is_pad_row || indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;
            if (renormalize) {
                selected_sum += inputs_after_softmax[thread_read_offset + expert];
            }
        }
        __syncthreads();
    }

    
    if (threadIdx.x == 0) {
        float scale = static_cast<float>(routed_scaling_factor);
        if (renormalize) {
            const float denom = selected_sum > 0.f ? selected_sum : 1.f;
            scale /= denom;
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = output[idx] * scale;
        }
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename IndType,
          typename InputType = float, ScoringFunc SF>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGating(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
        const float* bias, const double routed_scaling_factor, const bool* is_padding)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

    
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    

    
    
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    
    
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    
    
    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    
    float row_chunk[VPT];

    
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else { 
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
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

    if constexpr (SF == SCORING_SOFTMAX) {
      
      float thread_max = row_chunk[0];
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
      }


#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
      }

      
      
      float row_sum = 0;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }


#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
      }

      
      
      
      
      
      const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
      }
    } else if constexpr (SF == SCORING_SIGMOID) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
      }
    }

    
    
    
    
    
    
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (isnan(row_chunk[ii]) || isinf(row_chunk[ii])) {
        row_chunk[ii] = 0.f;
      }
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    
    float row_chunk_for_choice[VPT];
    
    if (bias != nullptr) {
#pragma unroll
      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          const int expert = first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
          float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
          row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] = row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk_for_choice[ii] = row_chunk[ii];
      }
    }

    
    
    int start_col = first_elt_read_by_thread;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        
        float max_val_for_choice = row_chunk_for_choice[0];
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                
                
                if (val_for_choice > max_val_for_choice)
                {
                    max_val_for_choice = val_for_choice;
                    max_val = val;
                    expert = col + ii;
                }
            }
        }


#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max_for_choice = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, THREADS_PER_ROW);
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            
            if (other_max_for_choice > max_val_for_choice || (other_max_for_choice == max_val_for_choice && other_expert < expert))
            {
                max_val_for_choice = other_max_for_choice;
                max_val = other_max;
                expert = other_expert;
            }
        }

        
        if (thread_group_idx == 0)
        {
            
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const bool is_pad_row = is_padding != nullptr && is_padding[thread_row];

            
            
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = is_pad_row ? static_cast<IndType>(-1)
                                       : (should_process_row ? (expert - start_expert) : NUM_EXPERTS);
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                
                row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
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
}

namespace detail
{

template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename IndType, typename InputType, ScoringFunc SF>
void topkGatingLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
    const float* bias, const double routed_scaling_factor, cudaStream_t stream, const bool* is_padding)
{
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
    topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, IndType, InputType, SF><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize, bias, routed_scaling_factor, is_padding);
}

#ifndef USE_ROCM
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                   \
    static_assert(WARP_SIZE == 32,                                            \
                  "Unsupported warp size. Only 32 is supported for CUDA");    \
    topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, MAX_BYTES, \
                             IndType, InputType, SF>(                         \
        gating_output, nullptr, topk_weights, topk_indices,                   \
        token_expert_indices, num_tokens, topk, 0, num_experts, renormalize,  \
        bias, routed_scaling_factor, stream, is_padding);
#else
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                    \
    if (WARP_SIZE == 64) {                                                     \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 64, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, routed_scaling_factor, stream, is_padding);                                            \
    } else if (WARP_SIZE == 32) {                                              \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, routed_scaling_factor, stream, is_padding);                                            \
    } else {                                                                   \
      assert(false &&                                                          \
             "Unsupported warp size. Only 32 and 64 are supported for ROCm");  \
    }
#endif


}
}



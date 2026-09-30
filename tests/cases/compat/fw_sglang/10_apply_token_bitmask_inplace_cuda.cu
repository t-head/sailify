


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#if !defined(USE_ROCM) && (!defined(CUDA_VERSION) || CUDA_VERSION < 12040)
#else

#ifndef CUDART_INF_FP16
#ifndef USE_ROCM
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#endif

#ifndef CUDART_INF_BF16
#ifndef USE_ROCM
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

constexpr int32_t BITS_PER_BLOCK = 32;
constexpr int32_t THREADS_PER_THREAD_BLOCK = 256;

template <typename T>
__device__ T NegativeInfinity() {
  return -INFINITY;
}

template <>
__device__ __half NegativeInfinity<__half>() {
#ifdef USE_ROCM
  return __float2half(-INFINITY);
#else
  return -CUDART_INF_FP16;
#endif
}

template <>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
#ifdef USE_ROCM
  return __nv_bfloat16(-INFINITY);
#else
  return -CUDART_INF_BF16;
#endif
}

template <typename T, typename PackedT>
__device__ PackedT PackedNegativeInfinity() {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  T packed[kAlignment];
#pragma unroll
  for (int i = 0; i < kAlignment; i++) {
    packed[i] = NegativeInfinity<T>();
  }
  return *reinterpret_cast<PackedT*>(packed);
}

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(THREADS_PER_THREAD_BLOCK) LogitsBitmaskKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

  const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  const int block_offset = blockIdx.x * THREADS_PER_THREAD_BLOCK * kBitsPerThread;
  T* logits_gmem_ptr = logits + batch_idx * logits_stride + block_offset;
  const int32_t* bitmask_gmem_ptr = bitmask + batch_idx * bitmask_stride + block_offset / BITS_PER_BLOCK;
  const int bitmask_inner_idx = threadIdx.x % (BITS_PER_BLOCK / kAlignment);
  T logits_reg[kAlignment];

#pragma unroll
  for (int offset = threadIdx.x * kAlignment; offset < THREADS_PER_THREAD_BLOCK * kBitsPerThread;
       offset += THREADS_PER_THREAD_BLOCK * kAlignment) {
    if (block_offset + offset >= vocab_size) {
      break;
    }

    const uint32_t bitmask_val =
        (~bitmask_gmem_ptr[offset / BITS_PER_BLOCK] >> (bitmask_inner_idx * kAlignment)) & kPackedMask;

    if (bitmask_val == 0) {
      continue;
    }

    if (bitmask_val == kPackedMask) {
      *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = PackedNegativeInfinity<T, PackedT>();
      continue;
    }

    *reinterpret_cast<PackedT*>(logits_reg) = *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset);
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
      if (((bitmask_val >> i) & 1)) {
        logits_reg[i] = NegativeInfinity<T>();
      }
    }
    *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = *reinterpret_cast<PackedT*>(logits_reg);
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr auto CeilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

#endif


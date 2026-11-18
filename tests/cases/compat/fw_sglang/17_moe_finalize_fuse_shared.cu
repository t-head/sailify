#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>


#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

namespace ffi = tvm::ffi;

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_uint16 = DLDataType{kDLUInt, 16, 1};
constexpr DLDataType dl_uint32 = DLDataType{kDLUInt, 32, 1};
constexpr DLDataType dl_uint64 = DLDataType{kDLUInt, 64, 1};
constexpr DLDataType dl_int8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType dl_int16 = DLDataType{kDLInt, 16, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType dl_float8_e5m2 = DLDataType{kDLFloat8_e5m2, 8, 1};
constexpr DLDataType dl_float4_e2m1fn = DLDataType{kDLFloat4_e2m1fn, 4, 1};
constexpr DLDataType dl_float4_e2m1fn_x2 = DLDataType{kDLFloat4_e2m1fn, 4, 2};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_bool = DLDataType{kDLBool, 8, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t uint8_code = encode_dlpack_dtype(dl_uint8);
constexpr int64_t int32_code = encode_dlpack_dtype(dl_int32);
constexpr int64_t int64_code = encode_dlpack_dtype(dl_int64);
constexpr int64_t float8_e4m3fn_code = encode_dlpack_dtype(dl_float8_e4m3fn);
constexpr int64_t float8_e5m2_code = encode_dlpack_dtype(dl_float8_e5m2);
constexpr int64_t float4_e2m1fn_code = encode_dlpack_dtype(dl_float4_e2m1fn);

constexpr DLDevice cpu = DLDevice{kDLCPU, 0};

#define CHECK_CUDA(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA) << #x " must be a CUDA tensor";
#define CHECK_CPU(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCPU) << #x " must be a host tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TVM_FFI_ICHECK_EQ(x.stride(-1), 1) \
  #x "must be contiguous at last dimension";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_CUDA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";
#define CHECK_DEVICE(a, b)                                           \
  TVM_FFI_ICHECK_EQ(a.device().device_type, b.device().device_type); \
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);


#include <cuda_runtime.h>

namespace sglang {

using BF16 = cutlass::bfloat16_t;

constexpr int FINALIZE_THREADS_PER_BLOCK = 256;
constexpr int MAX_TOPK = 64;


template <typename TypeExpW>
__global__ void moeFinalizeKernel(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* __restrict__ inPtr,
    int const* __restrict__ expandedIdxToPermutedIdx,
    TypeExpW const* __restrict__ expertWeightsPtr,
    BF16 const* __restrict__ sharedBiasPtr,
    BF16* __restrict__ outPtr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  for (int64_t tokenIdx = blockIdx.y; tokenIdx < numTokens; tokenIdx += gridDim.y) {
    for (int64_t hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      for (int k = 0; k < topK; k++) {
        int64_t const expandedIdx = tokenIdx * topK + k;
        int64_t const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale = static_cast<float>(expertWeightsPtr[expandedIdx]);
        float const val = static_cast<float>(inPtr[permutedIdx * hiddenDimPadded + hiddenIdx]);
        acc += scale * val;
      }
      if (sharedBiasPtr != nullptr) {
        acc += static_cast<float>(sharedBiasPtr[tokenIdx * hiddenDim + hiddenIdx]);
      }
      outPtr[tokenIdx * hiddenDim + hiddenIdx] = static_cast<BF16>(acc);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


__device__ inline float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

template <int TopKUnrollFactor>
struct IdxPackedTraits;
template <>
struct IdxPackedTraits<1> {
  using Packed = int;
};
template <>
struct IdxPackedTraits<2> {
  using Packed = int2;
};
template <>
struct IdxPackedTraits<4> {
  using Packed = int4;
};

template <typename TypeExpW, int TopKUnrollFactor>
__global__ void moeFinalizeKernelVecLoad(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* __restrict__ inPtr,
    int const* __restrict__ expandedIdxToPermutedIdx,
    TypeExpW const* __restrict__ expertWeightsPtr,
    BF16 const* __restrict__ sharedBiasPtr,
    BF16* __restrict__ outPtr) {
  static_assert(
      TopKUnrollFactor == 1 || TopKUnrollFactor == 2 || TopKUnrollFactor == 4, "TopKUnrollFactor must be 1, 2, or 4");
  using IdxPackedType = typename IdxPackedTraits<TopKUnrollFactor>::Packed;
  using IdxArrayType = cutlass::Array<int, TopKUnrollFactor>;
  using ScaleArrayType = cutlass::Array<TypeExpW, TopKUnrollFactor>;

  
  constexpr int FINALIZE_ELEM_PER_THREAD = 8;
  using InputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = hiddenDim / FINALIZE_ELEM_PER_THREAD;

  
  __shared__ ScaleArrayType scaleArrSmem[MAX_TOPK / TopKUnrollFactor];
  __shared__ IdxArrayType permutedIdxArrSmem[MAX_TOPK / TopKUnrollFactor];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx += blockDim.x) {
    int64_t const expandedIdx = tokenIdx * topK + kChunkIdx * TopKUnrollFactor;
    auto const permutedIdxPacked =
        reinterpret_cast<IdxPackedType const*>(expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    permutedIdxArrSmem[kChunkIdx] = *reinterpret_cast<IdxArrayType const*>(&permutedIdxPacked);
#pragma unroll
    for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
      scaleArrSmem[kChunkIdx][ki] = expertWeightsPtr[expandedIdx + ki];
    }
  }

  BF16* outputPtr = outPtr + tokenIdx * hiddenDim;
  auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(inPtr);
  auto const* sharedElemPtr =
      sharedBiasPtr != nullptr ? reinterpret_cast<InputElem const*>(sharedBiasPtr + tokenIdx * hiddenDim) : nullptr;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  __syncthreads();

  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;

    for (int kChunkIdx = 0; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx++) {
      IdxArrayType permutedIdxArr = permutedIdxArrSmem[kChunkIdx];
      InputElem inputElemArr[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;
        float4 input = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const*>(&input);
      }
      ScaleArrayType scaleArr = scaleArrSmem[kChunkIdx];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale = static_cast<float>(scaleArr[ki]);
        cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
        ComputeElem expertResult = toFloat(inputElemArr[ki]);
#pragma unroll
        for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
          threadOutput[e] += scale * expertResult[e];
        }
      }
    }

    if (sharedElemPtr != nullptr) {
      float4 shared = vectorizedLoadPtx(reinterpret_cast<float4 const*>(&sharedElemPtr[elemIndex]));
      InputElem sharedElem = *reinterpret_cast<InputElem const*>(&shared);
      cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
      ComputeElem sharedFloat = toFloat(sharedElem);
#pragma unroll
      for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
        threadOutput[e] += sharedFloat[e];
      }
    }

    cutlass::NumericArrayConverter<BF16, float, FINALIZE_ELEM_PER_THREAD> toBF16;
    outElemPtr[elemIndex] = toBF16(threadOutput);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename TypeExpW>
void dispatchFinalize(
    int numTokens,
    int hiddenDim,
    int hiddenDimPadded,
    int topK,
    BF16 const* inPtr,
    int const* expandedIdxPtr,
    void const* weightsPtrVoid,
    BF16 const* sharedPtr,
    BF16* outPtr,
    bool useVecLoad,
    cudaStream_t stream,
    cudaLaunchAttribute const* attrs,
    int numAttrs) {
  auto const* weightsPtr = static_cast<TypeExpW const*>(weightsPtrVoid);
  constexpr int kNumThreads = 256;

  if (!useVecLoad) {
    int const numBlocksX = (hiddenDim + kNumThreads - 1) / kNumThreads;
    int const numBlocksY = std::min(8192, numTokens);
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numBlocksX, numBlocksY);
    config.blockDim = dim3(kNumThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);

    cudaLaunchKernelEx(
        &config,
        moeFinalizeKernel<TypeExpW>,
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        topK,
        inPtr,
        expandedIdxPtr,
        weightsPtr,
        sharedPtr,
        outPtr);
    return;
  }

  auto launch = [&](auto unroll_tag) {
    constexpr int UNROLL = decltype(unroll_tag)::value;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numTokens);
    config.blockDim = dim3(FINALIZE_THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);
    cudaLaunchKernelEx(
        &config,
        moeFinalizeKernelVecLoad<TypeExpW, UNROLL>,
        numTokens,
        hiddenDim,
        hiddenDimPadded,
        topK,
        inPtr,
        expandedIdxPtr,
        weightsPtr,
        sharedPtr,
        outPtr);
  };
  
  if (topK % 4 == 0) {
    launch(std::integral_constant<int, 4>{});
  } else if (topK % 2 == 0) {
    launch(std::integral_constant<int, 2>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}



}

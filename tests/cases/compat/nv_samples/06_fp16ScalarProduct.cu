#include <cuda_runtime.h>


#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda_fp16.h"


#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif


#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API

static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef _CUFFT_H_

static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";

    case CUFFT_MISSING_DEPENDENCY:
      return "CUFFT_MISSING_DEPENDENCY";

    case CUFFT_NVRTC_FAILURE:
      return "CUFFT_NVRTC_FAILURE";

    case CUFFT_NVJITLINK_FAILURE:
      return "CUFFT_NVJITLINK_FAILURE";

    case CUFFT_NVSHMEM_FAILURE:
      return "CUFFT_NVSHMEM_FAILURE";
  }

  return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI

static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
  switch (error) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_

static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef CURAND_H_

static const char *_cudaGetErrorEnum(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NVJPEGAPI

static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
  switch (error) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H

static const char *_cudaGetErrorEnum(NppStatus error) {
  switch (error) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";

    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_BAD_ARG_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFF_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECT_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUAD_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEM_ALLOC_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_INPUT:
      return "NPP_INVALID_INPUT";

    case NPP_POINTER_ERROR:
      return "NPP_POINTER_ERROR";

    case NPP_WARNING:
      return "NPP_WARNING";

    case NPP_ODD_ROI_WARNING:
      return "NPP_ODD_ROI_WARNING";
#else

    
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";

    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";

    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";

    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";

    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";

    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";

    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_MEMFREE_ERR:
      return "NPP_MEMFREE_ERR";

    case NPP_MEMSET_ERR:
      return "NPP_MEMSET_ERR";

    case NPP_MEMCPY_ERR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERR:
      return "NPP_MIRROR_FLIP_ERR";
#else

    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";

    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";

    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";
#endif

    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";

    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";

    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";

    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";

    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";

    case NPP_ERROR:
      return "NPP_ERROR";

    case NPP_SUCCESS:
      return "NPP_SUCCESS";

    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";

    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";

    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
    
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";

    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";

    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";

    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";

    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";

    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";

    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";

    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";

    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";

    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";

    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";

    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";

    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";

    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";

    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";

    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";

    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";

    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";

    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";

    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";

    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";

    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";

    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";

    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";

    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
    
    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";

    case NPP_CORRUPTED_DATA_ERROR:
      return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#ifdef __DRIVER_TYPES_H__


#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
  }
}
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif


inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}


inline int _ConvertSMVer2Cores(int major, int minor) {
  
  
  typedef struct {
    int SM;  
    
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xa3, 128},
      {0xa7, 128},
      {0xb0, 128},
      {0xc0, 128},
      {0xc1, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  
  
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  
  
  typedef struct {
    int SM;  
    
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {0xa0, "Blackwell"},
      {0xa1, "Blackwell"},
      {0xa3, "Blackwell"},
      {0xa7, "Rubin"},
      {0xb0, "Blackwell"},
      {0xc0, "Blackwell"},
      {0xc1, "Blackwell"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  
  
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
  

#ifdef __CUDA_RUNTIME_H__
#endif

  

#endif  


#define NUM_OF_BLOCKS  128
#define NUM_OF_THREADS 128

__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v)
{
    if (threadIdx.x < 64)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
    __syncthreads();
    if (threadIdx.x < 32)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
    __syncthreads();
    if (threadIdx.x < 16)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
    __syncthreads();
    if (threadIdx.x < 8)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
    __syncthreads();
    if (threadIdx.x < 4)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
    __syncthreads();
    if (threadIdx.x < 2)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
    __syncthreads();
    if (threadIdx.x < 1)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
    __syncthreads();
}

__forceinline__ __device__ void reduceInShared_native(half2 *const v)
{
    if (threadIdx.x < 64)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
    __syncthreads();
    if (threadIdx.x < 16)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
    __syncthreads();
    if (threadIdx.x < 8)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
    __syncthreads();
    if (threadIdx.x < 4)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
    __syncthreads();
    if (threadIdx.x < 2)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
    __syncthreads();
    if (threadIdx.x < 1)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
    __syncthreads();
}

__global__ void
scalarProductKernel_intrinsics(half2 const *const a, half2 const *const b, float *const results, size_t const size)
{
    const int        stride = gridDim.x * blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    shArray[threadIdx.x] = __float2half2_rn(0.f);
    half2 value          = __float2half2_rn(0.f);

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
        value = __hfma2(a[i], b[i], value);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_intrinsics(shArray);

    if (threadIdx.x == 0) {
        half2 result        = shArray[0];
        float f_result      = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

__global__ void
scalarProductKernel_native(half2 const *const a, half2 const *const b, float *const results, size_t const size)
{
    const int        stride = gridDim.x * blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value(0.f, 0.f);
    shArray[threadIdx.x] = value;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
        value = a[i] * b[i] + value;
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_native(shArray);

    if (threadIdx.x == 0) {
        half2 result        = shArray[0];
        float f_result      = (float)result.y + (float)result.x;
        results[blockIdx.x] = f_result;
    }
}

void generateInput(half2 *a, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        half2 temp;
        temp.x = static_cast<float>(rand() % 4);
        temp.y = static_cast<float>(rand() % 2);
        a[i]   = temp;
    }
}

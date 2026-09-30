


#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#pragma once


#ifdef TORCH_TARGET_VERSION
#else
#endif


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

#include <cassert>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
#endif


#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDART_VERSION) && CUDART_VERSION >= 12090
  #define VLLM_256B_PTX_ENABLED 1
#else
  #define VLLM_256B_PTX_ENABLED 0
#endif

namespace vllm {


struct alignas(32) u32x8_t {
  uint32_t d[8];
};


template <bool support_256>
struct VecTraits;

template <>
struct VecTraits<true> {
  static constexpr int ARCH_MAX_VEC_SIZE = 32;
  using vec_t = u32x8_t;
};

template <>
struct VecTraits<false> {
  static constexpr int ARCH_MAX_VEC_SIZE = 16;
  using vec_t = int4;
};


template <typename T>
struct PackedTypeConverter {
  static_assert(sizeof(T) == 0,
                "PackedTypeConverter is not specialized for this type.");
};

template <>
struct PackedTypeConverter<half2> {
  using Type = half;
};

template <>
struct PackedTypeConverter<half> {
  using Type = half2;
};

template <>
struct PackedTypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct PackedTypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <>
struct PackedTypeConverter<float> {
  using Type = float2;
};

template <>
struct PackedTypeConverter<float2> {
  using Type = float;
};


template <typename T>
struct CUDATypeConverter {
  using Type = T;
};

template <>
struct CUDATypeConverter<__nv_half> {
  using Type = half;
};

template <>
struct CUDATypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat16;
};


template <class Type, bool use_256b>
struct alignas(VecTraits<use_256b>::ARCH_MAX_VEC_SIZE) PackedVec {
  static constexpr int NUM_ELTS =
      VecTraits<use_256b>::ARCH_MAX_VEC_SIZE /
      sizeof(typename PackedTypeConverter<Type>::Type);
  typename PackedTypeConverter<Type>::Type elts[NUM_ELTS];
};


__device__ __forceinline__ void ld256(u32x8_t& val, const u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(ptr));
#else
  assert(false && "ld256 requires SM100+ with CUDA 12.9+");
#endif
}

__device__ __forceinline__ void st256(u32x8_t& val, u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :
               : "l"(ptr), "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]),
                 "r"(val.d[3]), "r"(val.d[4]), "r"(val.d[5]), "r"(val.d[6]),
                 "r"(val.d[7])
               : "memory");
#else
  assert(false && "st256 requires SM100+ with CUDA 12.9+");
#endif
}


template <typename T>
__device__ __forceinline__ void ld256(T& val, const T* ptr) {
  static_assert(sizeof(T) == 32, "ld256 requires a 32-byte type");
  ld256(reinterpret_cast<u32x8_t&>(val), reinterpret_cast<const u32x8_t*>(ptr));
}

template <typename T>
__device__ __forceinline__ void st256(T& val, T* ptr) {
  static_assert(sizeof(T) == 32, "st256 requires a 32-byte type");
  st256(reinterpret_cast<u32x8_t&>(val), reinterpret_cast<u32x8_t*>(ptr));
}


template <typename T>
__device__ __forceinline__ void ld128(T& val, const T* ptr) {
  static_assert(sizeof(T) == 16, "ld128 requires a 16-byte type");
  *reinterpret_cast<int4*>(&val) = __ldg(reinterpret_cast<const int4*>(ptr));
}

template <typename T>
__device__ __forceinline__ void st128(T& val, T* ptr) {
  static_assert(sizeof(T) == 16, "st128 requires a 16-byte type");
  *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<int4*>(&val);
}


__forceinline__ __device__ u32x8_t ld256_cs(const u32x8_t* addr) {
#if VLLM_256B_PTX_ENABLED
  u32x8_t val;
  asm volatile("ld.global.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(addr));
  return val;
#else
  assert(false && "ld256_cs requires SM100+ with CUDA 12.9+");
  return u32x8_t{};
#endif
}

__forceinline__ __device__ void st256_cs(u32x8_t* addr, u32x8_t val) {
#if VLLM_256B_PTX_ENABLED
  asm volatile(
      "st.global.cs.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(addr),
      "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]), "r"(val.d[3]), "r"(val.d[4]),
      "r"(val.d[5]), "r"(val.d[6]), "r"(val.d[7]));
#else
  assert(false && "st256_cs requires SM100+ with CUDA 12.9+");
#endif
}


__device__ __forceinline__ int ld32(const int* addr) { return __ldg(addr); }

__device__ __forceinline__ void st32(int* addr, int val) { *addr = val; }


__forceinline__ __device__ int ld32_cs(const int* addr) {
  int val;
#ifndef USE_ROCM
  asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(addr));
#else
  val = ld32(addr);
#endif
  return val;
}

__forceinline__ __device__ void st32_cs(int* addr, int val) {
#ifndef USE_ROCM
  asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(addr), "r"(val));
#else
  st32(addr, val);
#endif
}


__forceinline__ __device__ int4 ld128_cs(const int4* addr) {
  int4 val;
#ifndef USE_ROCM
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
#else
  ld128(val, addr);
#endif
  return val;
}

__forceinline__ __device__ void st128_cs(int4* addr, int4 val) {
#ifndef USE_ROCM
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  st128(val, addr);
#endif
}


__device__ __forceinline__ void ld256_cg_or_zero(u32x8_t& val, const void* ptr,
                                                 bool pred) {
#if VLLM_256B_PTX_ENABLED
  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %8, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  mov.u32 %4, 0;\n"
      "  mov.u32 %5, 0;\n"
      "  mov.u32 %6, 0;\n"
      "  mov.u32 %7, 0;\n"
      "  @pr ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
      "}\n"
      : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
        "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
      : "r"((int)pred), "l"(ptr));
#else
  assert(false && "ld256_cg_or_zero requires SM100+ with CUDA 12.9+");
#endif
}

__device__ __forceinline__ void ld128_cg_or_zero(uint4& val, const void* ptr,
                                                 bool pred) {
#ifndef USE_ROCM
  uint32_t r0, r1, r2, r3;

  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %4, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  @pr ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"((int)pred), "l"(ptr));

  val = uint4{r0, r1, r2, r3};
#else
  assert(false && "ld128_cg_or_zero is not supported on ROCm");
#endif
}


__host__ __device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

__host__ __device__ __forceinline__ bool is_32byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0;
}


template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __bfloat1622float2(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __half22float2(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __float22bfloat162_rn(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __float22half2_rn(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t& x,
                                               const packed_t& y) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162> ||
                std::is_same_v<packed_t, __half2>) {
    return __hmul2(x, y);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return make_float2(x.x * y.x, x.y * y.y);
  }
}

}


#pragma once

#include <stdio.h>

#if defined(__HIPCC__)
  #define HOST_DEVICE_INLINE __host__ __device__
  #define DEVICE_INLINE __device__
  #define HOST_INLINE __host__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define DEVICE_INLINE __device__ __forceinline__
  #define HOST_INLINE __host__ __forceinline__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

namespace cuda_utils {

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

};  
#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>


#ifndef VLLM_LAUNCH_BLOCKS_CAP
  #define VLLM_LAUNCH_BLOCKS_CAP 4
#endif


#ifndef VLLM_MAX_THREADS_PER_SM
  #ifdef __CUDA_ARCH__

    
    #if (__CUDA_ARCH__ == 750)
      #define VLLM_MAX_THREADS_PER_SM 1024

    

    #elif (__CUDA_ARCH__ == 860) || (__CUDA_ARCH__ == 870) || \
        (__CUDA_ARCH__ == 890) || (__CUDA_ARCH__ == 1010) ||  \
        (__CUDA_ARCH__ == 1100) || (__CUDA_ARCH__ == 1200) || \
        (__CUDA_ARCH__ == 1210)
      #define VLLM_MAX_THREADS_PER_SM 1536

    

    #elif (__CUDA_ARCH__ == 700) || (__CUDA_ARCH__ == 720) || \
        (__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900) ||   \
        (__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1030)
      #define VLLM_MAX_THREADS_PER_SM 2048

    
    #else
      #define VLLM_MAX_THREADS_PER_SM 2048
    #endif

  #else
  
    #define VLLM_MAX_THREADS_PER_SM 2048
  #endif
#endif


#define VLLM_BLOCKS_DIV(VAL) (VLLM_MAX_THREADS_PER_SM / (VAL))
#define VLLM_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0)                       \
       ? 1                            \
       : (((VAL) < VLLM_LAUNCH_BLOCKS_CAP) ? (VAL) : VLLM_LAUNCH_BLOCKS_CAP))
#define VLLM_BLOCKS_PER_SM(BLOCK_THREADS) \
  VLLM_CLAMP_BLOCKS_PER_SM(VLLM_BLOCKS_DIV(BLOCK_THREADS))


static inline int vllm_runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = VLLM_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return VLLM_CLAMP_BLOCKS_PER_SM(blocks);
}


#define NVFP4_ENABLE_ELTS16 1


#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <utility>


#if defined(NVFP4_ENABLE_ELTS16) && defined(CUDART_VERSION) && \
    CUDART_VERSION >= 12090
  #define ELTS_PER_THREAD 16
  #define CVT_FP4_PACK16 1
constexpr int CVT_FP4_ELTS_PER_THREAD = 16;
#else
  #define ELTS_PER_THREAD 8
  #define CVT_FP4_PACK16 0
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
#endif

constexpr int CVT_FP4_SF_VEC_SIZE = 16;

namespace vllm {

template <typename Int>
__host__ __device__ inline Int round_up(Int x, Int y) {
  static_assert(std::is_integral_v<Int>,
                "round_up argument must be integral type");
  return ((x + y - 1) / y) * y;
}

template <typename Int>
__host__ __device__ __forceinline__ Int div_round_up(Int x, Int y) {
  return (x + y - 1) / y;
}


inline int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}


inline std::pair<int64_t, int64_t> computeSwizzledSFShape(int64_t m,
                                                          int64_t n) {
  int64_t rounded_m = round_up(m, static_cast<int64_t>(128));
  int64_t scale_n = n / CVT_FP4_SF_VEC_SIZE;
  int64_t rounded_n = round_up(scale_n, static_cast<int64_t>(4));
  return {rounded_m, rounded_n / 4};
}


inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}


__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

struct u32x2 {
  uint32_t lo, hi;
};

using fp4_packed_t = std::conditional_t<CVT_FP4_PACK16, u32x2, uint32_t>;

__device__ __forceinline__ u32x2 fp32_vec16_to_e2m1(float2 (&array)[8]) {
  u32x2 out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      ".reg .b8 b4;\n"
      ".reg .b8 b5;\n"
      ".reg .b8 b6;\n"
      ".reg .b8 b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b0,  %3,  %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b1,  %5,  %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b2,  %7,  %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b3,  %9,  %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}\n"
      : "=r"(out.lo), "=r"(out.hi)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return out;
}

__device__ __forceinline__ uint32_t pack_fp4(float2 (&v)[4]) {
  return fp32_vec8_to_e2m1(v);
}

__device__ __forceinline__ u32x2 pack_fp4(float2 (&v)[8]) {
  return fp32_vec16_to_e2m1(v);
}


__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}


template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int32_t numKTiles, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  
  
  
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF != 0) {
    return nullptr;
  }

  
  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  
  
  int32_t mTileIdx = mIdx >> 7;         
  int32_t outerMIdx = mIdx & 31;        
  int32_t innerMIdx = (mIdx >> 5) & 3;  
  int32_t kTileIdx = kIdx >> 2;         
  int32_t innerKIdx = kIdx & 3;         

  
  
  
  int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                         << 9 |
                     (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}

template <class SFType>
__device__ __forceinline__ uint8_t* sf_out_rowmajor_u8(int row, int pack,
                                                       int packs_per_row_sf,
                                                       SFType* SFout) {
  constexpr int PACK = CVT_FP4_ELTS_PER_THREAD;
  constexpr int THREADS_PER_SF =
      CVT_FP4_SF_VEC_SIZE / PACK;  

  if (threadIdx.x % THREADS_PER_SF != 0) return nullptr;

  int sf_col =
      pack / THREADS_PER_SF;  
  int64_t off = (int64_t)row * packs_per_row_sf + sf_col;

  return (uint8_t*)SFout + off;
}


template <class Type, int CVT_FP4_NUM_THREADS_PER_SF, bool UE8M0_SF = false>
__device__ __forceinline__ fp4_packed_t cvt_warp_fp16_to_fp4(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float SFScaleVal, uint8_t* SFout) {
  
  auto localMax = __habs2(vec.elts[0]);

  
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  
  float vecMax = float(__hmax(localMax.x, localMax.y));

  
  float SFValue;
  uint8_t fp8SFVal;

  if constexpr (UE8M0_SF) {
    
    
    
    
    uint32_t max_bits = __float_as_uint(vecMax);
    
    
    
    uint32_t rounded_bits = (max_bits + (1u << 21)) & 0xFF800000u;
    uint32_t biased_exp = (rounded_bits >> 23) & 0xFFu;
    uint32_t scale_exp = (biased_exp > 2u) ? (biased_exp - 2u) : 0u;
    scale_exp = min(scale_exp, 254u);
    fp8SFVal = static_cast<uint8_t>(scale_exp);
    
    uint32_t sf_bits = scale_exp << 23;
    SFValue = __uint_as_float(sf_bits);
  } else {
    
    SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    
    SFValue = float(tmp);
  }

  
  if (SFout) *SFout = fp8SFVal;

  
  
  
  
  float outputScale;
  if constexpr (UE8M0_SF) {
    
    outputScale = SFValue != 0.0f ? (1.0f / SFValue) : 0.0f;
  } else {
    
    outputScale = SFValue != 0.0f
                      ? reciprocal_approximate_ftz(
                            SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
  }

  
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  
  return pack_fp4(fp2Vals);
}


__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + __expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

template <class Type>
__inline__ __device__ PackedVec<Type, CVT_FP4_PACK16> compute_silu_mul(
    const PackedVec<Type, CVT_FP4_PACK16>& x_vec,
    const PackedVec<Type, CVT_FP4_PACK16>& y_vec) {
  PackedVec<Type, CVT_FP4_PACK16> result;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    
    using packed_t = typename PackedTypeConverter<Type>::Type;
    float2 silu_vec = silu2(cast_to_float2(x_vec.elts[i]));
    float2 y_f2 = cast_to_float2(y_vec.elts[i]);
    result.elts[i] = cast_to_packed<packed_t>(
        make_float2(silu_vec.x * y_f2.x, silu_vec.y * y_f2.y));
  }
  return result;
}

}


namespace vllm {


template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t outputCols,
                    int32_t num_padded_cols, Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  
  int32_t const numKTiles = (outputCols + 63) / 64;

  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  
  
  
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  
  
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      
      bool valid_input = (rowIdx < numRows) && (elem_idx < numCols);
      bool valid_output = (rowIdx < numRows) && (elem_idx < outputCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid_input);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid_input);
      }

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      if (valid_output) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (outputCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          int64_t outOffset =
              rowIdx * (outputCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
          out[outOffset] = out_val;
        }
      }
    }
  }
}


template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_sf_major(int32_t numRows, int32_t numCols,
                             int32_t outputCols, int32_t sf_n_unpadded,
                             int32_t num_packed_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  
  
  
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  
  
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    if (colIdx < num_packed_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      
      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      auto sf_out =
          sf_out_rowmajor_u8<uint32_t>(rowIdx, colIdx, sf_n_unpadded, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      if (rowIdx < numRows) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (outputCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          int64_t outOffset =
              rowIdx * (outputCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
          out[outOffset] = out_val;
        }
      }
    }
  }
}

}


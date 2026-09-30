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

#ifndef CONCAT_MLA_Q_CUH_
#define CONCAT_MLA_Q_CUH_


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


namespace vllm {


template <typename DType, int NOPE_DIM>
__global__ void ConcatMLAQKernel(
    DType* __restrict__ q_out, const DType* __restrict__ ql_nope,
    const DType* __restrict__ q_pe, const int num_tokens, const int num_heads,
    const int64_t out_stride_0, const int64_t out_stride_1,
    const int64_t nope_stride_0, const int64_t nope_stride_1,
    const int64_t pe_stride_0, const int64_t pe_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= num_tokens * num_heads) return;

  const int token_id = flat_warp_id / num_heads;
  const int head_id = flat_warp_id % num_heads;
  const int lane_id = threadIdx.x & 31;

  constexpr bool use_256b = VLLM_256B_PTX_ENABLED;
  constexpr int nope_vec_loads =
      NOPE_DIM * sizeof(DType) / (VecTraits<use_256b>::ARCH_MAX_VEC_SIZE * 32);

  const DType* nope_src =
      ql_nope + token_id * nope_stride_0 + head_id * nope_stride_1;
  DType* nope_dst = q_out + token_id * out_stride_0 + head_id * out_stride_1;

#pragma unroll
  for (int i = 0; i < nope_vec_loads; i++) {
    const int offset = i * 32 + lane_id;
    if constexpr (use_256b) {
      st256_cs(reinterpret_cast<u32x8_t*>(nope_dst) + offset,
               ld256_cs(reinterpret_cast<const u32x8_t*>(nope_src) + offset));
    } else {
      st128_cs(reinterpret_cast<int4*>(nope_dst) + offset,
               ld128_cs(reinterpret_cast<const int4*>(nope_src) + offset));
    }
  }

  const int* rope_src = reinterpret_cast<const int*>(
      q_pe + token_id * pe_stride_0 + head_id * pe_stride_1);
  int* rope_dst = reinterpret_cast<int*>(q_out + token_id * out_stride_0 +
                                         head_id * out_stride_1 + NOPE_DIM);

  st32_cs(rope_dst + lane_id, ld32_cs(rope_src + lane_id));
}

}

#endif  


#ifdef USE_ROCM
#pragma once
#include <hip/hip_fp8.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

#pragma once


#pragma once

#include <stdint.h>

namespace vllm {


template <typename T, int VEC_SIZE>
struct Vec {};


template <typename T>
struct FloatVec {};


template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <typename T>
inline __device__ float sum(T v);

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

}


#pragma once


#pragma once


#include <stdint.h>

namespace vllm {


struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};


template <>
struct Vec<float, 1> {
  using Type = float;
};
template <>
struct Vec<float, 2> {
  using Type = float2;
};
template <>
struct Vec<float, 4> {
  using Type = float4;
};


template <>
struct FloatVec<float> {
  using Type = float;
};
template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};


inline __device__ float add(float a, float b) { return a + b; }

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}


template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}


inline __device__ float fma(float a, float b, float c) { return a * b + c; }

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}


template <>
inline __device__ float sum(float v) {
  return v;
}

template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template <>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}


inline __device__ float dot(float a, float b) { return a * b; }

inline __device__ float dot(float2 a, float2 b) {
  float2 c = mul<float2, float2, float2>(a, b);
  return c.x + c.y;
}

inline __device__ float dot(Float4_ a, Float4_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  return acc.x + acc.y;
}

inline __device__ float dot(Float8_ a, Float8_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
  return acc.x + acc.y;
}


inline __device__ void from_float(float& dst, float src) { dst = src; }

inline __device__ void from_float(float2& dst, float2 src) { dst = src; }

inline __device__ void from_float(float4& dst, float4 src) { dst = src; }


inline __device__ float to_float(float u) { return u; }

inline __device__ float2 to_float(float2 u) { return u; }

inline __device__ float4 to_float(float4 u) { return u; }

inline __device__ Float4_ to_float(Float4_ u) { return u; }

inline __device__ Float8_ to_float(Float8_ u) { return u; }


inline __device__ void zero(float& dst) { dst = 0.f; }

}


#ifdef USE_ROCM
  #include <hip/hip_fp16.h>
#endif

#include <stdint.h>

namespace vllm {


template <>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template <>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template <>
struct Vec<uint16_t, 4> {
  using Type = uint2;
};
template <>
struct Vec<uint16_t, 8> {
  using Type = uint4;
};


template <>
struct FloatVec<uint16_t> {
  using Type = float;
};
template <>
struct FloatVec<uint32_t> {
  using Type = float2;
};
template <>
struct FloatVec<uint2> {
  using Type = Float4_;
};
template <>
struct FloatVec<uint4> {
  using Type = Float8_;
};


inline __device__ uint32_t h0_h0(uint16_t a) {
#ifndef USE_ROCM
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
  return b;
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u16[0] = a;
  tmp.u16[1] = a;
  return tmp.u32;
#endif
}

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
#ifndef USE_ROCM
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return make_float2(half_to_float(lo), half_to_float(hi));
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  float2 ret;
  ret.x = half_to_float(tmp.u16[0]);
  ret.y = half_to_float(tmp.u16[1]);
  return ret;
#endif
}

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ uint32_t float2_to_half2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
               : "=r"(tmp.u32)
               : "f"(f.y), "f"(f.x));
  #else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
  #endif
#else
  tmp.u16[0] = float_to_half(f.x);
  tmp.u16[1] = float_to_half(f.y);
#endif
  return tmp.u32;
}


inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}


template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template <>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  return c;
}

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

template <>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
  return fc;
}


inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
#ifndef USE_ROCM
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
#else
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n"
               : "=v"(d)
               : "v"(a), "v"(b), "v"(c));
#endif
  return d;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}


template <>
inline __device__ float sum(uint16_t v) {
  return half_to_float(v);
}

template <>
inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

template <>
inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

template <>
inline __device__ float sum(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}


inline __device__ void from_float(uint16_t& dst, float src) {
  dst = float_to_half(src);
}

inline __device__ void from_float(uint32_t& dst, float2 src) {
  dst = float2_to_half2(src);
}

inline __device__ void from_float(uint2& dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(uint4& dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}


inline __device__ float to_float(uint16_t u) { return half_to_float(u); }

inline __device__ float2 to_float(uint32_t u) { return half2_to_float2(u); }

inline __device__ Float4_ to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}


inline __device__ void zero(uint16_t& dst) { dst = uint16_t(0); }

}


#pragma once


#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;
#endif

#include <stdint.h>

namespace vllm {


struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};


template <>
struct Vec<__nv_bfloat16, 1> {
  using Type = __nv_bfloat16;
};
template <>
struct Vec<__nv_bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <>
struct Vec<__nv_bfloat16, 4> {
  using Type = bf16_4_t;
};


template <>
struct FloatVec<__nv_bfloat16> {
  using Type = float;
};
template <>
struct FloatVec<__nv_bfloat162> {
  using Type = float2;
};
template <>
struct FloatVec<bf16_4_t> {
  using Type = Float4_;
};


inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat1622float2(val);
#endif
  __builtin_unreachable();  
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat162bfloat162(val);
#endif
  __builtin_unreachable();  
}


inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  #ifndef USE_ROCM
  return a + b;
  #else
  return __hadd(a, b);
  #endif
#endif
  __builtin_unreachable();  
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hadd2(a, b);
#endif
  __builtin_unreachable();  
}

inline __device__ bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float2 add(__nv_bfloat162 a, float2 fb) {
  float2 fa = bf1622float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}


template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hmul(a, b);
#endif
  __builtin_unreachable();  
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hmul2(a, b);
#endif
  __builtin_unreachable();  
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return c;
}

template <>
inline __device__ bf16_4_t mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return c;
}

template <>
inline __device__ float mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  float fa = __bfloat162float(a);
  float fb = __bfloat162float(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return fc;
}


inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hfma2(a, b, c);
#endif
  __builtin_unreachable();  
}

inline __device__ __nv_bfloat162 fma(__nv_bfloat16 a, __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hfma2(bf162bf162(a), b, c);
#endif
  __builtin_unreachable();  
}

inline __device__ bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ bf16_4_t fma(__nv_bfloat16 a, bf16_4_t b, bf16_4_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float fc) {
  return __bfloat162float(a) * __bfloat162float(b) + fc;
}

inline __device__ float2 fma(__nv_bfloat162 a, __nv_bfloat162 b, float2 fc) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(__nv_bfloat16 a, __nv_bfloat162 b, float2 fc) {
  return fma(bf162bf162(a), b, fc);
}

inline __device__ Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(__nv_bfloat16 a, bf16_4_t b, Float4_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}


template <>
inline __device__ float sum(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
inline __device__ float sum(__nv_bfloat162 v) {
  float2 vf = bf1622float2(v);
  return vf.x + vf.y;
}

template <>
inline __device__ float sum(bf16_4_t v) {
  return sum(v.x) + sum(v.y);
}


inline __device__ void from_float(__nv_bfloat16& dst, float src) {
  dst = __float2bfloat16(src);
}

inline __device__ void from_float(__nv_bfloat162& dst, float2 src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst = __float22bfloat162_rn(src);
#endif
}

inline __device__ void from_float(bf16_4_t& dst, Float4_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#endif
}


inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}


inline __device__ void zero(__nv_bfloat16& dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  
  dst = __ushort_as_bfloat16((unsigned short)0x0000U);
#endif
}

}

#pragma once


#include <stdint.h>
#ifdef ENABLE_FP8
  #ifndef USE_ROCM
    #include <cuda_fp8.h>
  #endif  
#endif    

namespace vllm {

template <>
struct Vec<uint8_t, 1> {
  using Type = uint8_t;
};

template <>
struct Vec<uint8_t, 2> {
  using Type = uint16_t;
};

template <>
struct Vec<uint8_t, 4> {
  using Type = uint32_t;
};

template <>
struct Vec<uint8_t, 8> {
  using Type = uint2;
};

}


namespace vllm {
#ifdef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8


template <typename fp8_type>
__device__ __forceinline__ fp8_type cvt_c10(float const r) {
  return {};
}


template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  return x;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale) {
  return x;
}

    #if HIP_FP8_TYPE_OCP
using fp8_type = __hip_fp8_e4m3;
using fp8x2_type = __hip_fp8x2_e4m3;
    #else
using fp8_type = __hip_fp8_e4m3_fnuz;
using fp8x2_type = __hip_fp8x2_e4m3_fnuz;
    #endif


template <>
__inline__ __device__ uint16_t
vec_conversion<uint16_t, uint8_t>(const uint8_t& a) {
  return __hip_cvt_fp8_to_halfraw(a, fp8_type::__default_interpret).x;
}


template <>
__inline__ __device__ uint32_t
vec_conversion<uint32_t, uint16_t>(const uint16_t& a) {
  union {
    __half2_raw h2r;
    uint32_t ui32;
  } tmp;
  tmp.h2r = __hip_cvt_fp8x2_to_halfraw2(a, fp8_type::__default_interpret);
  return tmp.ui32;
}


template <>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
  tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
  return tmp.u32x2;
}


template <>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x);
  tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y);
  return tmp.u64x2;
}

using __nv_bfloat16 = __hip_bfloat16;


template <>
__inline__ __device__ __nv_bfloat16
vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a) {
  fp8_type f8;
  f8.__x = a;
  return __float2bfloat16(static_cast<float>(f8));
}

using __nv_bfloat162 = __hip_bfloat162;


template <>
__inline__ __device__ __nv_bfloat162
vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a) {
  __nv_bfloat162 res;
  res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a);
  res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U));
  return res;
}


template <>
__inline__ __device__ bf16_4_t
vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a) {
  bf16_4_t res;
  res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a);
  res.y = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U));
  return res;
}


template <>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a) {
  fp8_type f8;
  f8.__x = a;
  return static_cast<float>(f8);
}


template <>
__inline__ __device__ float2
vec_conversion<float2, uint16_t>(const uint16_t& a) {
  fp8x2_type f8x2;
  f8x2.__x = a;
  return static_cast<float2>(f8x2);
}


template <>
__inline__ __device__ Float4_
vec_conversion<Float4_, uint32_t>(const uint32_t& a) {
  Float4_ res;
  res.x = vec_conversion<float2, uint16_t>((uint16_t)a);
  res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U));
  return res;
}


template <>
__inline__ __device__ float4
vec_conversion<float4, uint32_t>(const uint32_t& a) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}


template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a) {
  Float4_ tmp1, tmp2;
  tmp1 = vec_conversion<Float4_, uint32_t>(a.x);
  tmp2 = vec_conversion<Float4_, uint32_t>(a.y);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}


template <>
__inline__ __device__ uint8_t
vec_conversion<uint8_t, uint16_t>(const uint16_t& a) {
  __half_raw tmp;
  tmp.x = a;
  return __hip_cvt_halfraw_to_fp8(tmp, fp8_type::__default_saturation,
                                  fp8_type::__default_interpret);
}

template <>
__inline__ __device__ uint16_t
vec_conversion<uint16_t, uint32_t>(const uint32_t& a) {
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;
  return __hip_cvt_halfraw2_to_fp8x2(tmp.h2r, fp8_type::__default_saturation,
                                     fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint8_t
vec_conversion<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a) {
  return __hip_cvt_float_to_fp8(__bfloat162float(a),
                                fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(const float& a) {
  return __hip_cvt_float_to_fp8(a, fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint32_t
vec_conversion<uint32_t, float2>(const float2& a) {
  union {
    half2 float16;
    uint32_t uint32;
  };

  float16 = __float22half2_rn(a);
  return uint32;
}


template <>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a) {
  uint2 b;
  float2 val;
  val.x = a.x.x;
  val.y = a.x.y;
  b.x = vec_conversion<uint32_t, float2>(val);

  val.x = a.y.x;
  val.y = a.y.y;
  b.y = vec_conversion<uint32_t, float2>(val);
  return b;
}


template <>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a) {
  float4 b;
  b.x = a.x.x;
  b.y = a.x.y;
  b.z = a.y.x;
  b.w = a.y.y;
  return b;
}


template <>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a) {
  uint4 b;
  b.x = vec_conversion<uint32_t, float2>(a.x);
  b.y = vec_conversion<uint32_t, float2>(a.y);
  b.z = vec_conversion<uint32_t, float2>(a.z);
  b.w = vec_conversion<uint32_t, float2>(a.w);
  return b;
}


template <>
__inline__ __device__ __nv_bfloat162
vec_conversion<__nv_bfloat162, float2>(const float2& a) {
  __nv_bfloat162 b = __float22bfloat162_rn(a);
  return b;
}


template <>
__inline__ __device__ bf16_4_t
vec_conversion<bf16_4_t, Float4_>(const Float4_& a) {
  bf16_4_t b;
  b.x = __float22bfloat162_rn(a.x);
  b.y = __float22bfloat162_rn(a.y);
  return b;
}


using __nv_bfloat16 = __hip_bfloat16;


template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a, float scale) {
  fp8_type f8;
  f8.__x = a;
  return __float2bfloat16(static_cast<float>(f8) * scale);
}


template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a,
                                                float scale) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale);
  res.y =
      scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), scale);
  return res;
}


template <>
__inline__ __device__ bf16_4_t
scaled_vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a, float scale) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale);
  return res;
}


template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, float scale) {
  fp8_type f8;
  f8.__x = a;
  return static_cast<float>(f8) * scale;
}


template <>
__inline__ __device__ float2
scaled_vec_conversion<float2, uint16_t>(const uint16_t& a, float scale) {
  fp8x2_type f8x2;
  f8x2.__x = a;
  return static_cast<float2>(f8x2) * scale;
}


template <>
__inline__ __device__ Float4_
scaled_vec_conversion<Float4_, uint32_t>(const uint32_t& a, const float scale) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale);
  return res;
}


template <>
__inline__ __device__ float4
scaled_vec_conversion<float4, uint32_t>(const uint32_t& a, float scale) {
  Float4_ res = scaled_vec_conversion<Float4_, uint32_t>(a, scale);
  return {res.x.x, res.x.y, res.y.x, res.y.y};
}


template <>
__inline__ __device__ Float8_
scaled_vec_conversion<Float8_, uint2>(const uint2& a, float scale) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}


template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint8_t>(const uint8_t& a, float scale) {
  __half_raw res;
  res.data = scaled_vec_conversion<float, uint8_t>(a, scale);
  return res.x;
}


template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t& a, float scale) {
  union {
    __half2_raw h2r;
    uint32_t ui32;
  } tmp;
  tmp.h2r = __hip_cvt_fp8x2_to_halfraw2(a, fp8_type::__default_interpret);
  tmp.h2r.x.data *= scale;
  tmp.h2r.y.data *= scale;
  return tmp.ui32;
}


template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, uint32_t>(const uint32_t& a, float scale) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale);
  tmp.u32[1] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), scale);
  return tmp.u32x2;
}


template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2& a,
                                                                float scale) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale);
  return tmp.u64x2;
}


template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, uint16_t>(const uint16_t& a, float scale) {
  __half_raw tmp;
  tmp.x = a;
  tmp.data /= scale;
  return __hip_cvt_halfraw_to_fp8(tmp, fp8_type::__default_saturation,
                                  fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint32_t>(const uint32_t& a, float scale) {
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;
  tmp.h2r.x.data /= scale;
  tmp.h2r.y.data /= scale;
  return __hip_cvt_halfraw2_to_fp8x2(tmp.h2r, fp8_type::__default_saturation,
                                     fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint2>(const uint2& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, uint32_t>(a.x, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, uint32_t>(a.y, scale);
  return tmp.ui32;
}


template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint4>(const uint4& a,
                                                                float scale) {
  union {
    uint2 ui2[2];
    uint4 ui4;
  } tmp;
  tmp.ui4 = a;
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[0], scale);
  res.y = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[1], scale);
  return res;
}


template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, float scale) {
  return __hip_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, __nv_bfloat162>(
    const __nv_bfloat162& a, float scale) {
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.x, scale);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.y, scale);
  return tmp.ui16;
}


template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, bf16_4_t>(const bf16_4_t& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.x, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.y, scale);
  return tmp.ui32;
}


template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, float>(const float& a, float scale) {
  return __hip_cvt_float_to_fp8(a / scale, fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, float2>(const float2& a, float scale) {
  return __hip_cvt_float2_to_fp8x2(a / scale, fp8_type::__default_saturation,
                                   fp8_type::__default_interpret);
}


template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, float4>(const float4& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, float2>({a.x, a.y}, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, float2>({a.z, a.w}, scale);
  return tmp.ui32;
}
  #endif  

  
  
  
  

}
#endif  
}

#else
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include <cuda.h>
#endif

#if defined(__gfx942__)
constexpr float kFp8ScaleDivisor = 224.f;
#else
constexpr float kFp8ScaleDivisor = 448.f;
#endif


namespace {


constexpr int64_t kRocmDefaultMaxBatchDescriptors = 8192;

constexpr const char* kMaxBatchDescriptorsEnv =
    "VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS";


int64_t resolve_max_batch_descriptors() {
  static const int64_t cached = []() -> int64_t {
    const char* val = std::getenv(kMaxBatchDescriptorsEnv);
    const int64_t override_val = val ? std::atoll(val) : 0;
    if (override_val > 0) return override_val;
#if defined(USE_ROCM)
    return kRocmDefaultMaxBatchDescriptors;
#else
    return 0;
#endif
  }();
  return cached;
}
}  


#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                     \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                   \
      <<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<KV_T*>(key.data_ptr()),                          \
          reinterpret_cast<KV_T*>(value.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                 \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),               \
          slot_mapping.const_data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                              \
          reinterpret_cast<const float*>(k_scale.data_ptr()),               \
          reinterpret_cast<const float*>(v_scale.data_ptr()));


#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)                \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>              \
      <<<grid, block, 0, stream>>>(                                          \
          reinterpret_cast<KV_T*>(key.data_ptr()),                           \
          reinterpret_cast<KV_T*>(value.data_ptr()),                         \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, page_stride, \
          head_stride, key_stride, value_stride, num_heads, head_size,       \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()),    \
          reinterpret_cast<const float*>(v_scale.data_ptr()),                \
          kv_scale_stride);


#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)                    \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                  \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                           \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                           \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,         \
          reinterpret_cast<const float*>(scale.data_ptr()));


#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)                 \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>               \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                           \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                           \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,         \
          reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);


#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ)            \
  vllm::gather_and_maybe_dequant_cache_page<SCALAR_T, CACHE_T, KV_DTYPE,    \
                                            ENTRY_SZ>                       \
      <<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                 \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                      \
          block_table.const_data_ptr<int32_t>(),                            \
          cu_seq_lens.const_data_ptr<int32_t>(), num_reqs,                  \
          static_cast<int32_t>(num_tokens), block_size, block_table_stride, \
          cache_block_stride, cache_entry_stride, dst_entry_stride,         \
          reinterpret_cast<const float*>(scale.data_ptr()), seq_starts_ptr)

#define CALL_GATHER_CACHE_576(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 576)

#define CALL_GATHER_CACHE_320(SCALAR_T, CACHE_T, KV_DTYPE) \


#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(k.data_ptr()),                              \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_block_stride, use_ue8m0);


#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                    \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                  \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,                 \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),                \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                                 \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                       \
          reinterpret_cast<char*>(dst_k.data_ptr()),                          \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                      \
          block_table.const_data_ptr<int32_t>(),                              \
          cu_seq_lens.const_data_ptr<int32_t>(), batch_size, dst_k.stride(0), \
          dst_k.size(1), kv_cache.stride(0), kv_cache.stride(1),              \
          kv_cache.size(1), block_table.size(1), num_tokens,                  \
          quant_block_size);






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
#pragma once

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

struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
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
struct Vec<__nv_bfloat16, 8> {
  using Type = bf16_8_t;
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
template <>
struct FloatVec<bf16_8_t> {
  using Type = Float8_;
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

inline __device__ bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
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

inline __device__ Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
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
inline __device__ bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return c;
}

template <>
inline __device__ bf16_8_t mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.w);
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

template <>
inline __device__ Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);
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

inline __device__ bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ bf16_8_t fma(__nv_bfloat16 a, bf16_8_t b, bf16_8_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
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

inline __device__ Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(__nv_bfloat16 a, bf16_8_t b, Float8_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
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

template <>
inline __device__ float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
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

inline __device__ void from_float(bf16_8_t& dst, Float8_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
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

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};


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


#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {
#ifndef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8


template <typename>
inline constexpr bool _no_conversion_specialization = false;

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(
    const Tin& x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
  static_assert(_no_conversion_specialization<Tin>,
                "no vec_conversion specialization for this (Tout, Tin) pair");
}


template <>
__inline__ __device__ __nv_fp8_e4m3
vec_conversion<__nv_fp8_e4m3, float>(
    const float& a, const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<__nv_fp8_e4m3>(a);
    #else
  return __nv_fp8_e4m3(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            __nv_fp8_e4m3::from_bits());
    #endif
}


template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  static_assert(
      _no_conversion_specialization<Tin>,
      "no scaled_vec_conversion specialization for this (Tout, Tin) pair");
}


template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return float_to_half(half_to_float(tmp.x) * scale);
}


template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
  tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
  return tmp.u32;
}


template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U),
                                                         scale, fp8_type);
  return tmp.u32x2;
}


template <>
__inline__ __device__ uint4
scaled_vec_conversion<uint4, uint2>(const uint2& a, const float scale,
                                    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
  return tmp.u64x2;
}


template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  
  
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp * scale);
}


template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale,
                                                        fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                        scale, fp8_type);
  return res;
}


template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale,
                                                          fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, fp8_type);
  return res;
}


template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}


template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  uint16_t tmp = res.x;

  
  return half_to_float(tmp) * scale;
}


template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  
  uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
  
  return half2_to_float2(tmp);
}


template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale,
                                                  fp8_type);
  return res;
}


template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}


template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}


template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  
}


template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return scaled_vec_conversion<uint8_t, __nv_bfloat16>(
      reinterpret_cast<const __nv_bfloat16&>(a), scale, fp8_type);
}

template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_half>(
    const __nv_half& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return scaled_vec_conversion<uint8_t, uint16_t>(
      reinterpret_cast<const uint16_t&>(a), scale, fp8_type);
}


template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}


template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}
  #endif  

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  #if 0  
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return vec_conversion<Tout, Tin>(x, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return vec_conversion<Tout, Tin>(x, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  
}

  
  
  
  

}
#endif  
}

#else
  #include <hip/hip_fp8.h>
#endif
#include <cuda_runtime.h>
#include <cfloat>
#include <type_traits>

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
namespace kimi_k3_fused_ops {

namespace {
inline int getSMVersion() {
}
}


constexpr int kKvLoraRank = 512;                             
constexpr int kQkNopeHeadDim = 128;                          
constexpr int kQkRopeHeadDim = 64;                           
constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  
constexpr int kVHeadDim = 128;                               
constexpr int kCacheEntry = kKvLoraRank + kQkRopeHeadDim;    
constexpr int kVecElems = 8;  

#if defined(USE_ROCM) && defined(__gfx942__)
constexpr float kFp8Max = 224.0f;
#else
constexpr float kFp8Max = 448.0f;
#endif


constexpr float kFp8ScaleDivisor = kFp8Max;


template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void copyChunk8(void* dst, const scalar_t* src,
                                           float scale_inv,
                                           const float* cos_sin = nullptr,
                                           int rope_elem_base = 0) {
  uint4 const v = *reinterpret_cast<const uint4*>(src);
  if constexpr (FP8 || APPLY_ROPE) {
    using Converter = vllm::_typeConvert<scalar_t>;
    if constexpr (!Converter::exists) {
      return;
    } else {
      auto const* p =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v);
      float f[kVecElems];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 x = Converter::convert(p[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
      }
      if constexpr (APPLY_ROPE) {
#pragma unroll
        for (int i = 0; i < kVecElems / 2; i++) {
          int const pair_idx = rope_elem_base / 2 + i;
          float const cos = static_cast<float>(cos_sin[pair_idx]);
          float const sin =
              static_cast<float>(cos_sin[pair_idx + kQkRopeHeadDim / 2]);
          float const x = f[2 * i];
          float const y = f[2 * i + 1];
          f[2 * i] = x * cos - y * sin;
          f[2 * i + 1] = x * sin + y * cos;
        }
      }
      if constexpr (!FP8) {
        uint4 out;
        auto* o = reinterpret_cast<typename Converter::packed_hip_type*>(&out);
#pragma unroll
        for (int i = 0; i < kVecElems / 2; i++) {
          o[i] = Converter::convert(make_float2(f[2 * i], f[2 * i + 1]));
        }
        *reinterpret_cast<uint4*>(dst) = out;
        return;
      }
#ifndef USE_ROCM
      uint2 out;
      auto* o2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 s = make_float2(f[2 * i] * scale_inv, f[2 * i + 1] * scale_inv);
        s.x = fminf(fmaxf(s.x, -kFp8Max), kFp8Max);
        s.y = fminf(fmaxf(s.y, -kFp8Max), kFp8Max);
        o2[i] = __nv_cvt_float2_to_fp8x2(s, __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(dst) = out;
#else
      uint8_t out[kVecElems];
  #pragma unroll
      for (int i = 0; i < kVecElems; i++) {
        float s = fminf(fmaxf(f[i] * scale_inv, -kFp8Max), kFp8Max);
        out[i] = rocm_cvt_float_to_fp8_e4m3(s);
      }
      *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(out);
#endif
    }
  } else {
    *reinterpret_cast<uint4*>(dst) = v;
  }
}


template <typename scalar_t>
__device__ __forceinline__ void copyChunk8UnitFp8(uint8_t* dst,
                                                  const scalar_t* src) {
  using Converter = vllm::_typeConvert<scalar_t>;
  if constexpr (!Converter::exists) {
    return;
  } else {
#ifndef USE_ROCM
    uint4 const input = *reinterpret_cast<const uint4*>(src);
    auto const* input2 =
        reinterpret_cast<typename Converter::packed_hip_type const*>(&input);
    uint2 output;
    auto* output2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&output);
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
        output2[i] = __nv_cvt_bfloat16raw2_to_fp8x2(
            static_cast<__nv_bfloat162_raw>(input2[i]), __NV_SATFINITE,
            __NV_E4M3);
      } else {
        output2[i] = __nv_cvt_halfraw2_to_fp8x2(
            static_cast<__half2_raw>(input2[i]), __NV_SATFINITE, __NV_E4M3);
      }
    }
    *reinterpret_cast<uint2*>(dst) = output;
#else
    copyChunk8<scalar_t, true>(dst, src, 1.0f);
#endif
  }
}


template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeFullKey(void* dst, const scalar_t* k_nope,
                                             const scalar_t* k_pe, int laneId,
                                             int dst_elem_size, float scale_inv,
                                             const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kQkHeadDim; e += 32 * kVecElems) {
    if (e < kQkNopeHeadDim) {
      copyChunk8<scalar_t, FP8>(d + e * dst_elem_size, k_nope + e, scale_inv);
    } else {
      int const rope_e = e - kQkNopeHeadDim;
      copyChunk8<scalar_t, FP8, APPLY_ROPE>(
          d + e * dst_elem_size, k_pe + rope_e, scale_inv, cos_sin, rope_e);
    }
  }
}


template <typename scalar_t, bool FP8, bool KPE_FP8>
__device__ __forceinline__ void writeFullKeyPack(void* dst,
                                                 const scalar_t* k_nope,
                                                 const void* k_pe, int laneId) {
  static_assert(!KPE_FP8 || FP8, "an fp8 k_pe requires an fp8 key output");
  constexpr int kElemSize = FP8 ? 1 : sizeof(scalar_t);
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kQkHeadDim; e += 16 * kVecElems) {
    void* out = d + e * kElemSize;
    const scalar_t* src = k_nope + e;
    if (e >= kQkNopeHeadDim) {
      int const rope_e = e - kQkNopeHeadDim;
      if constexpr (KPE_FP8) {
        *reinterpret_cast<uint2*>(out) = *reinterpret_cast<const uint2*>(
            reinterpret_cast<const uint8_t*>(k_pe) + rope_e);
        continue;
      }
      src = reinterpret_cast<const scalar_t*>(k_pe) + rope_e;
    }
    if constexpr (FP8) {
      copyChunk8UnitFp8(reinterpret_cast<uint8_t*>(out), src);
    } else {
      *reinterpret_cast<uint4*>(out) = *reinterpret_cast<const uint4*>(src);
    }
  }
}


template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writePrefillQuery(
    void* dst, const scalar_t* q, int laneId, int dst_elem_size,
    float scale_inv, const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  if constexpr (FP8) {
    for (int e = laneId * kVecElems; e < kQkHeadDim; e += 32 * kVecElems) {
      if (e < kQkNopeHeadDim) {
        copyChunk8<scalar_t, true>(d + e * dst_elem_size, q + e, scale_inv);
      } else {
        int const rope_e = e - kQkNopeHeadDim;
        copyChunk8<scalar_t, true, APPLY_ROPE>(d + e * dst_elem_size, q + e,
                                               scale_inv, cos_sin, rope_e);
      }
    }
  } else if constexpr (APPLY_ROPE) {
    for (int e = laneId * kVecElems; e < kQkRopeHeadDim; e += 32 * kVecElems) {
      copyChunk8<scalar_t, false, true>(
          d + (kQkNopeHeadDim + e) * dst_elem_size, q + kQkNopeHeadDim + e,
          scale_inv, cos_sin, e);
    }
  }
}


template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeLatent576(void* dst, const scalar_t* a512,
                                               const scalar_t* b64, int laneId,
                                               int dst_elem_size,
                                               float scale_inv,
                                               const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kCacheEntry; e += 32 * kVecElems) {
    if (e < kKvLoraRank) {
      copyChunk8<scalar_t, FP8>(d + e * dst_elem_size, a512 + e, scale_inv);
    } else {
      int const rope_e = e - kKvLoraRank;
      copyChunk8<scalar_t, FP8, APPLY_ROPE>(d + e * dst_elem_size, b64 + rope_e,
                                            scale_inv, cos_sin, rope_e);
    }
  }
}


template <typename scalar_t, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeDsMlaCache(
    uint8_t* row, const scalar_t* kvc, const scalar_t* pe, int laneId,
    const float* cos_sin = nullptr) {
  constexpr int kElemsPerLane = kKvLoraRank / 32;  
  int const tile = laneId >> 3;                    
  scalar_t vals[kElemsPerLane];
  *reinterpret_cast<uint4*>(vals) =
      *reinterpret_cast<const uint4*>(kvc + laneId * kElemsPerLane);
  *reinterpret_cast<uint4*>(vals + 8) =
      *reinterpret_cast<const uint4*>(kvc + laneId * kElemsPerLane + 8);

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    max_abs = fmaxf(max_abs, fabsf(static_cast<float>(vals[i])));
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 8));
  }
  float const tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);
  if ((laneId & 7) == 0) {
    reinterpret_cast<float*>(row)[kKvLoraRank / 4 + tile] = tile_scale;
  }
  uint8_t res[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    res[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }
  *reinterpret_cast<uint4*>(row + laneId * kElemsPerLane) =
      *reinterpret_cast<const uint4*>(res);
  scalar_t* row16 = reinterpret_cast<scalar_t*>(row);
  scalar_t* rope_dst = row16 + kKvLoraRank / 2 + 8 + laneId * 2;
  if constexpr (APPLY_ROPE) {
    using Converter = vllm::_typeConvert<scalar_t>;
    if constexpr (!Converter::exists) {
      return;
    } else {
      using packed_t = typename Converter::packed_hip_type;
      packed_t const src = *reinterpret_cast<packed_t const*>(pe + laneId * 2);
      float2 const xy = Converter::convert(src);
      float const cos = static_cast<float>(cos_sin[laneId]);
      float const sin =
          static_cast<float>(cos_sin[laneId + kQkRopeHeadDim / 2]);
      *reinterpret_cast<packed_t*>(rope_dst) = Converter::convert(
          make_float2(xy.x * cos - xy.y * sin, xy.x * sin + xy.y * cos));
    }
  } else {
    *reinterpret_cast<int32_t*>(rope_dst) =
        *reinterpret_cast<const int32_t*>(pe + laneId * 2);
  }
}


template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAKeyConcatKVCacheInsertKernel(
    scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    scalar_t* __restrict__ k_out, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, scalar_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache, int const num_tokens,
    int const num_heads, int const cache_block_size) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    scalar_t* qh = q + tokenIdx * q_tok_stride + slotIdx * q_head_stride;
    writePrefillQuery<scalar_t, false, APPLY_ROPE>(
        qh, qh, laneId, sizeof(scalar_t), 1.0f, rope_cache);
    writeFullKey<scalar_t, false, APPLY_ROPE>(
        k_out + tokenIdx * ko_tok_stride + slotIdx * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + slotIdx * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
        rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      scalar_t* row = k_cache +
                      (slot_id / cache_block_size) * cache_block_stride +
                      (slot_id % cache_block_size) * cache_token_stride;
      writeLatent576<scalar_t, false, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
          rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAQKVQuantKVCacheFp8Kernel(
    const scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ v, int64_t const v_tok_stride,
    int64_t const v_head_stride, uint8_t* __restrict__ q_fp8,
    int64_t const qo_tok_stride, int64_t const qo_head_stride,
    uint8_t* __restrict__ k_fp8, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ v_fp8,
    int64_t const vo_tok_stride, int64_t const vo_head_stride,
    uint8_t* __restrict__ k_cache, int64_t const cache_block_stride,
    int64_t const cache_token_stride, const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_inv,
    const float* __restrict__ k_scale_inv,
    const float* __restrict__ v_scale_inv,
    const float* __restrict__ cache_scale_inv, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    int const h = slotIdx;
    
    float const qsi = __ldg(q_scale_inv);
    const scalar_t* qh = q + tokenIdx * q_tok_stride + h * q_head_stride;
    uint8_t* qo = q_fp8 + tokenIdx * qo_tok_stride + h * qo_head_stride;
    writePrefillQuery<scalar_t, true, APPLY_ROPE>(qo, qh, laneId, 1, qsi,
                                                  rope_cache);
    
    writeFullKey<scalar_t, true, APPLY_ROPE>(
        k_fp8 + tokenIdx * ko_tok_stride + h * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + h * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, 1, __ldg(k_scale_inv),
        rope_cache);
    
    float const vsi = __ldg(v_scale_inv);
    const scalar_t* vh = v + tokenIdx * v_tok_stride + h * v_head_stride;
    uint8_t* vo = v_fp8 + tokenIdx * vo_tok_stride + h * vo_head_stride;
    for (int e = laneId * kVecElems; e < kVHeadDim; e += 32 * kVecElems) {
      copyChunk8<scalar_t, true>(vo + e, vh + e, vsi);
    }
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      
      
      float const ksi = __ldg(cache_scale_inv);
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeLatent576<scalar_t, true, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, 1, ksi, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename scalar_t, bool FP8, bool KPE_FP8>
__global__ void fusedKimiK3MLAKVConcatPackKernel(
    const scalar_t* __restrict__ k_nope, int64_t const kn_tok_stride,
    int64_t const kn_head_stride, const void* __restrict__ k_pe,
    int64_t const k_pe_tok_stride, const scalar_t* __restrict__ v,
    int64_t const v_tok_stride, int64_t const v_head_stride,
    void* __restrict__ k_out, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ v_fp8,
    int64_t const vo_tok_stride, int64_t const vo_head_stride,
    int const num_tokens, int const num_heads) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 16;
  int const rowInWarp = (threadIdx.x % 32) / 16;
  int64_t const globalWarpIdx =
      static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
  int64_t const totalRows = static_cast<int64_t>(num_tokens) * num_heads;
  if (globalWarpIdx * 2 >= totalRows) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  constexpr int kOutElemSize = FP8 ? 1 : sizeof(scalar_t);
  int64_t const gridStride =
      static_cast<int64_t>(gridDim.x) * warpsPerBlock * 2;
  for (int64_t row = globalWarpIdx * 2 + rowInWarp; row < totalRows;
       row += gridStride) {
    int const tokenIdx = static_cast<int>(row / num_heads);
    int const headIdx = static_cast<int>(row % num_heads);
    const void* pe;
    if constexpr (KPE_FP8) {
      pe = reinterpret_cast<const uint8_t*>(k_pe) + tokenIdx * k_pe_tok_stride;
    } else {
      pe = reinterpret_cast<const scalar_t*>(k_pe) + tokenIdx * k_pe_tok_stride;
    }
    writeFullKeyPack<scalar_t, FP8, KPE_FP8>(
        reinterpret_cast<uint8_t*>(k_out) +
            (tokenIdx * ko_tok_stride + headIdx * ko_head_stride) *
                kOutElemSize,
        k_nope + tokenIdx * kn_tok_stride + headIdx * kn_head_stride, pe,
        laneId);

    
    if constexpr (FP8) {
      const scalar_t* vh =
          v + tokenIdx * v_tok_stride + headIdx * v_head_stride;
      uint8_t* vo = v_fp8 + tokenIdx * vo_tok_stride + headIdx * vo_head_stride;
      for (int e = laneId * kVecElems; e < kVHeadDim; e += 16 * kVecElems) {
        copyChunk8UnitFp8(vo + e, vh + e);
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLAKeyConcatDsMlaInsertKernel(
    scalar_t* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const scalar_t* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    scalar_t* __restrict__ k_out, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    scalar_t* qh = q + tokenIdx * q_tok_stride + slotIdx * q_head_stride;
    writePrefillQuery<scalar_t, false, APPLY_ROPE>(
        qh, qh, laneId, sizeof(scalar_t), 1.0f, rope_cache);
    
    writeFullKey<scalar_t, false, APPLY_ROPE>(
        k_out + tokenIdx * ko_tok_stride + slotIdx * ko_head_stride,
        k_nope + tokenIdx * kn_tok_stride + slotIdx * kn_head_stride,
        k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
        rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeDsMlaCache<scalar_t, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename scalar_t, bool Q_FP8, bool KV_FP8, bool APPLY_ROPE>
__global__ void fusedKimiK3MLADecodeQConcatKVCacheKernel(
    const scalar_t* __restrict__ ql_nope, int64_t const qn_tok_stride,
    int64_t const qn_head_stride, const scalar_t* __restrict__ q_pe,
    int64_t const qpe_tok_stride, int64_t const qpe_head_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    void* __restrict__ mqa_q, int64_t const mq_tok_stride,
    int64_t const mq_head_stride, void* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_inv,
    const float* __restrict__ cache_scale_inv, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  constexpr int kMqElem = Q_FP8 ? 1 : sizeof(scalar_t);
  constexpr int kCacheElem = KV_FP8 ? 1 : sizeof(scalar_t);
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    float const qsi = Q_FP8 ? __ldg(q_scale_inv) : 1.0f;
    writeLatent576<scalar_t, Q_FP8, APPLY_ROPE>(
        reinterpret_cast<uint8_t*>(mqa_q) +
            (tokenIdx * mq_tok_stride + slotIdx * mq_head_stride) * kMqElem,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        kMqElem, qsi, rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      float const ksi = KV_FP8 ? __ldg(cache_scale_inv) : 1.0f;
      writeLatent576<scalar_t, KV_FP8, APPLY_ROPE>(
          reinterpret_cast<uint8_t*>(k_cache) +
              (slot_id / cache_block_size * cache_block_stride +
               slot_id % cache_block_size * cache_token_stride) *
                  kCacheElem,
          kv_c + tokenIdx * kv_c_tok_stride, k_pe + tokenIdx * k_pe_tok_stride,
          laneId, kCacheElem, ksi, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename scalar_t, bool APPLY_ROPE>
__global__ void fusedKimiK3MLADecodeQConcatDsMlaKernel(
    const scalar_t* __restrict__ ql_nope, int64_t const qn_tok_stride,
    int64_t const qn_head_stride, const scalar_t* __restrict__ q_pe,
    int64_t const qpe_tok_stride, int64_t const qpe_head_stride,
    const scalar_t* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const scalar_t* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    scalar_t* __restrict__ mqa_q, int64_t const mq_tok_stride,
    int64_t const mq_head_stride, uint8_t* __restrict__ k_cache,
    int64_t const cache_block_stride, int64_t const cache_token_stride,
    const int64_t* __restrict__ slot_mapping, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
  if constexpr (APPLY_ROPE) {
    rope_cache = cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim;
  }

  if (slotIdx < num_heads) {
    writeLatent576<scalar_t, false, APPLY_ROPE>(
        mqa_q + tokenIdx * mq_tok_stride + slotIdx * mq_head_stride,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        sizeof(scalar_t), 1.0f, rope_cache);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      writeDsMlaCache<scalar_t, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, rope_cache);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename KernelT, typename... Args>
static void launchPdlSlots(KernelT kernel, int num_tokens, int slots_per_token,
                           int rows_per_warp, int max_blocks,
                           cudaStream_t stream, Args... args) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_rows = static_cast<int64_t>(num_tokens) * slots_per_token;
  int64_t const total_warps = (total_rows + rows_per_warp - 1) / rows_per_warp;
  int grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (max_blocks > 0 && grid > max_blocks) grid = max_blocks;
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;
  cudaLaunchKernelEx(&config, kernel, args...);
#else
  
  
  
  
  kernel<<<grid, kBlockSize, 0, stream>>>(args...);
  
#endif
}


template <typename KernelT, typename... Args>
static void launchPdl(KernelT kernel, int num_tokens, int num_heads,
                      cudaStream_t stream, Args... args) {
  launchPdlSlots(kernel, num_tokens, num_heads + 1, 1, 0, stream, args...);
}


}
}


namespace {
}


namespace {


}


namespace {


}



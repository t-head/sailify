


#include <numeric>

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
#include <cstdlib>
#include <string>

namespace vllm {


inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    std::string env_key = "VLLM_BATCH_INVARIANT";
    const char* val = std::getenv(env_key.c_str());
    return (val && std::atoi(val) != 0) ? 1 : 0;
  }();
  return cached;
}

}

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

#pragma once


#include <cmath>

#ifndef USE_ROCM
#else
#endif


#include <cmath>


#ifdef TORCH_TARGET_VERSION
#else
  #ifdef USE_ROCM
  #endif
#endif

#ifndef USE_ROCM
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
#endif


namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

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


namespace vllm {


template <typename scalar_t, typename fp8_type, int VEC_SIZE>
__global__ void rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,          
    const scalar_t* __restrict__ input,  
    const int input_stride,
    const scalar_t* __restrict__ weight,  
    const float* __restrict__ scale,      
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const scalar_t* input_row = input + blockIdx.x * input_stride;

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  
  float const scale_inv = 1.0f / *scale;

  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  for (int idx = threadIdx.x; idx < hidden_size / VEC_SIZE; idx += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[idx];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[idx];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      float w = static_cast<float>(src2.val[j]);
      
      
      
      
      
      scalar_t out_norm = static_cast<scalar_t>(x * s_variance * w);
      out[blockIdx.x * hidden_size + idx * VEC_SIZE + j] =
          scaled_fp8_conversion<true, fp8_type>(static_cast<float>(out_norm),
                                                scale_inv);
    }
  }
}


template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    
    scalar_t* __restrict__ input,  
    const int input_stride,
    scalar_t* __restrict__ residual,      
    const scalar_t* __restrict__ weight,  
    const float* __restrict__ scale,      
    const float epsilon, const int num_tokens, const int hidden_size) {
  
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  


  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int stride_id = blockIdx.x * vec_input_stride + idx;
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[stride_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> w = weight_v[idx];
    using Converter = _typeConvert<scalar_t>;
    using HipT = typename Converter::hip_type;
#pragma unroll
    for (int i = 0; i < width; ++i) {
      float x = Converter::convert(res.data[i]);
      float wf = Converter::convert(w.data[i]);
      
      
      
      
      HipT out_norm_h = Converter::convert(x * s_variance * wf);
      out[id * width + i] = scaled_fp8_conversion<true, fp8_type>(
          Converter::convert(out_norm_h), scale_inv);
    }
  }
}


template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    
    scalar_t* __restrict__ input,  
    const int input_stride,
    scalar_t* __restrict__ residual,      
    const scalar_t* __restrict__ weight,  
    const float* __restrict__ scale,      
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    
    
    scalar_t out_norm = static_cast<scalar_t>(x * s_variance * w);
    out[blockIdx.x * hidden_size + idx] = scaled_fp8_conversion<true, fp8_type>(
        static_cast<float>(out_norm), scale_inv);
  }
}

}


#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                     \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                       \
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {    \
        VLLM_STABLE_DISPATCH_FP8_TYPES(                                      \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {   \
              vllm::fused_add_rms_norm_static_fp8_quant_kernel<scalar_t,     \
                                                               width, fp8_t> \
                  <<<grid, block, 0, stream>>>(                              \
                      out.mutable_data_ptr<fp8_t>(),                         \
                      input.mutable_data_ptr<scalar_t>(), input_stride,      \
                      residual.mutable_data_ptr<scalar_t>(),                 \
                      weight.const_data_ptr<scalar_t>(),                     \
                      scale.const_data_ptr<float>(), epsilon, num_tokens,    \
                      hidden_size);                                          \
            });                                                              \
      });

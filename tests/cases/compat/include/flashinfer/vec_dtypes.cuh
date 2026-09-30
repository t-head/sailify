#pragma once
#include <cstdint>

namespace flashinfer {

template <typename T, uint32_t VEC_SIZE>
struct vec_t {
  T val[VEC_SIZE];

  __device__ __host__ void fill(T v) {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) val[i] = v;
  }

  __device__ __host__ void load(const T* ptr) {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) val[i] = ptr[i];
  }

  __device__ __host__ void store(T* ptr) const {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) ptr[i] = val[i];
  }

  template <typename OtherT>
  __device__ __host__ void cast_load(const OtherT* ptr) {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) val[i] = static_cast<T>(ptr[i]);
  }

  template <typename OtherT>
  __device__ __host__ void cast_store(OtherT* ptr) const {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) ptr[i] = static_cast<OtherT>(val[i]);
  }

  __device__ __host__ T& operator[](uint32_t i) { return val[i]; }
  __device__ __host__ const T& operator[](uint32_t i) const { return val[i]; }
};

}

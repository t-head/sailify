#pragma once

namespace cutlass {

struct bfloat16_t {
  float v;

  bfloat16_t() = default;
  __device__ __host__ bfloat16_t(float f) : v(f) {}
  __device__ __host__ operator float() const { return v; }
};

struct half_t {
  float v;

  half_t() = default;
  __device__ __host__ half_t(float f) : v(f) {}
  __device__ __host__ operator float() const { return v; }
};

}

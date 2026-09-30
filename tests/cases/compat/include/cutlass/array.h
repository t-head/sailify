#pragma once

namespace cutlass {

template <typename T, int N>
struct Array {
  T storage[N];

  __device__ __host__ T& operator[](int i) { return storage[i]; }
  __device__ __host__ const T& operator[](int i) const { return storage[i]; }
};

}

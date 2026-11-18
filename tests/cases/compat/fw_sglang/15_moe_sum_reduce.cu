#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

union Pack16B {
  uint4 v;
  __nv_bfloat16 u16[8];
};

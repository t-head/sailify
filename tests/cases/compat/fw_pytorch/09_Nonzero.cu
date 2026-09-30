#include <cuda_runtime.h>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif

namespace at::native {

namespace {
template <typename T>
struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const T& a) const {
    return (a != T(0));
  }
};

__global__ void write_fill_value(int64_t * inp, int64_t * total, int64_t fill_value, int64_t n){
  int64_t total_val = *total;
                                     

  for (int64_t idx = total_val + (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
      inp[idx] = fill_value;
  }
}



}}
#include <cuda_runtime.h>
#include <vector>

template <int N>
struct InputArray {
  int values[N];
};

template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}

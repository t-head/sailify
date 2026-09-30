#include "test_buf.h"

__global__ void kernel_free_null(uint64_t *dp) {
  free(nullptr);
  free(NULL);
  free(0);
  free((void*)dp[0]);
}

#define CHECK_CUDA_CALL(v)         \
  do {                             \
    cudaError_t result = v;           \
    if (result != cudaSuccess) {  \
      std::cout << "CUDA func call failed with error " << cudaGetErrorString(result) << "\n"; \
      exit(-1); \
    } \
    std::cout << "CUDA func call " << #v << " succeeded !\n"; \
  } while(0)

TEST(heap_memory, free_null) {
  auto testDBuf = testBuf<uint64_t, memory_scope::device>(1);
  testDBuf.clear();
  kernel_free_null<<<1, 1>>>(testDBuf);
  CHECK_CUDA_CALL(cudaDeviceSynchronize());
}
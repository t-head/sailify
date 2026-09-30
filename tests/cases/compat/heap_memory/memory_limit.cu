#include "test_buf.h"

__global__ void kernel_memory_limit(uint8_t *dp, size_t size) {
  int tid = threadIdx.x;
  size_t heapsize = size / blockDim.x;
  uint8_t *heap = (uint8_t*)malloc(heapsize);
  memset(heap + heapsize - blockDim.x - 1, tid & 0xff, blockDim.x);
  dp[tid] = heap[heapsize - 1 - tid];
  free(heap);
}

bool test_heap_memory_limit(size_t size, int blocksize) {
  TLOG_INFO << "Running with memory limit: " << size << ", block size: " << blocksize << "\n";
  size_t cur_size = 0;
  cudaDeviceGetLimit(&cur_size, cudaLimitMallocHeapSize);
  TLOG_INFO << "Current heap limit: " << cur_size << "\n";
  size_t padded_size = size * 2;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, padded_size);
  cudaDeviceGetLimit(&cur_size, cudaLimitMallocHeapSize);
  TLOG_INFO << "After set, current heap limit: " << cur_size << "\n";
  auto testDBuf = testBuf<uint8_t, memory_scope::device>(blocksize);
  auto testHBuf = testBuf<uint8_t, memory_scope::host>(blocksize);
  testDBuf.clear();
  kernel_memory_limit<<<1, blocksize>>>(testDBuf, size);
  testHBuf = testDBuf;
  bool pass = true;
  testHBuf.foreach([&](uint8_t v, int i) {
    if (v != (i & 0xff)) {
      TLOG_INFO << "Expected value: " << (uint32_t)(i & 0xff) << ", actual: " << (uint32_t)v
        << ", index: " << i << "\n";
      pass = false;
      return;
    }
  });
  TLOG_INFO << "Running " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(heap_memory, memory_limit) {
  bool ret = test_heap_memory_limit(16 * 1024 * 1024, 1024);
  ret &= test_heap_memory_limit(5ull * 1024, 32);
  EXPECT_TRUE(ret);
}
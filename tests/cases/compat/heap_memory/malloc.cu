#include "test_buf.h"

__device__ void assign_data(uint8_t** arr, uint8_t *dp, size_t size) {
  uint8_t *sptr = arr[blockDim.x - 1 - threadIdx.x];
  memset(sptr + size / 4, (threadIdx.x & 0xff), size / 2);
  memcpy(dp + size * threadIdx.x, arr[threadIdx.x], size);
}

__global__ void kernel_malloc(uint8_t *dp, size_t size) {
  uint8_t *hptr = (uint8_t *)malloc(size);
  memset(hptr, 0xaa, size / 2);
  memset(hptr + size / 2, 0xbb, size / 2);
  __shared__ uint8_t** hptr_arr;
  if (threadIdx.x == 0) {
    hptr_arr = (uint8_t**)malloc(sizeof(uint8_t*) * blockDim.x);
  }
  __syncthreads();
  hptr_arr[threadIdx.x] = hptr;
  __syncthreads();
  assign_data(hptr_arr, dp, size);
  free(hptr);
  if (threadIdx.x == 0) {
    free(hptr_arr);
  }
}

TEST(heap_memory, malloc) {
  size_t blocksize = 128;
  size_t heapsize = 4;
  size_t bufsize = blocksize * heapsize;
  auto testDBuf = testBuf<uint8_t, memory_scope::device>(bufsize);
  auto testHBuf = testBuf<uint8_t, memory_scope::host>(bufsize);
  testDBuf.clear();
  kernel_malloc<<<1, blocksize>>>(testDBuf, heapsize);
  testHBuf = testDBuf;
  testHBuf.foreach([&](uint8_t v, int i) {
    int pos = i % 4;
    uint8_t rid = 127 - i / 4;
    uint8_t golden = (pos == 0) ? 0xaa : (pos == 3) ? 0xbb : rid;
    EXPECT_EQ(v, golden);
  });
}
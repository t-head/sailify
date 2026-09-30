#include "test_buf.h"

struct myStruct {
  int x;
  int y;
};

__global__ void kernel_aligned_malloc(uint8_t *ptr, size_t size, size_t align, int loop) {
  uint8_t *dptr;
  auto is_aligned = [&](uint8_t *p) {return ((uint64_t)(p) & (align - 1)) == 0;};
  for (int l = 1; l <= loop; l ++) {
    dptr = (uint8_t*)__nv_aligned_device_malloc(size, align);
    memset(dptr, (l & 0xff), size);
    if (!is_aligned(dptr)) {
      ptr[0] = uint8_t((uint64_t)(dptr) & 0xff);
      free(dptr);
      return;
    }
    memcpy(ptr, dptr, size);
    free(dptr); 
  }
}

bool test_heap_aligned_malloc(size_t size, size_t align, int loop) {
  TLOG_INFO << "Running with heapsize: " << size << ", alignment: " << align
    << ", loop: " << loop << "\n";
  size_t blocksize = 1;
  size_t bufsize = size * blocksize;
  auto testDBuf = testBuf<uint8_t, memory_scope::device>(bufsize);
  auto testHBuf = testBuf<uint8_t, memory_scope::host>(bufsize);
  testDBuf.clear();
  kernel_aligned_malloc<<<1, blocksize>>>(testDBuf, size, align, loop);
  testHBuf = testDBuf;
  bool pass = true;
  testHBuf.foreach([&](uint8_t v, int i) {
    if (v != (loop & 0xff)) {
      TLOG_INFO << "Expected value: " << (uint32_t)(loop & 0xff) << ", actual: " << (uint32_t)v << "\n";
      pass = false;
      return;
    }
  });
  TLOG_INFO << "Running " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(heap_memory, aligned_malloc) {
  bool ret = test_heap_aligned_malloc(16, 16, 32);
  ret &= test_heap_aligned_malloc(128, 128, 100);
  ret &= test_heap_aligned_malloc(4, 16, 50);
  EXPECT_TRUE(ret);
}
#include "test_buf.h"

struct S {
  unsigned x;
  unsigned y;
  unsigned z;
  unsigned w;
  __device__ S(uint4 data) : x{data.x}, y{data.y}, z{data.z}, w{data.w} {}
};
#include <stdint.h>
__global__ void kernel(uint8_t *dp) {
  S* sp = (S*)__builtin_assume_aligned(dp, 16);
  uint4 data = {1, 2, 3, 4};
  sp[0] = data;
}

TEST(compiler_hint, builtin_assume_aligned) {
  auto testDBuf = testBuf<uint32_t, memory_scope::device>(4);
  auto testHBuf = testBuf<uint32_t, memory_scope::host>(4);
  kernel<<<1, 1>>>((uint8_t*)testDBuf.get_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 1);
  EXPECT_EQ(testHBuf[1], 2);
  EXPECT_EQ(testHBuf[2], 3);
  EXPECT_EQ(testHBuf[3], 4);
}

__global__ void kernel2(uint8_t *dp) {
  int tid = threadIdx.x;
  uint8_t* p = (uint8_t*)__builtin_assume_aligned(dp, 16, 4);
  S* sp = (S*)(p - 4);
  uint4 data = {1, 2, 3, 4};
  sp[0] = data;
}

TEST(compiler_hint, builtin_assume_aligned2) {
  auto testDBuf = testBuf<uint32_t, memory_scope::device>(5);
  auto testHBuf = testBuf<uint32_t, memory_scope::host>(5);
  kernel2<<<1, 1>>>((uint8_t*)(testDBuf.get_pointer() + 1));
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 1);
  EXPECT_EQ(testHBuf[1], 2);
  EXPECT_EQ(testHBuf[2], 3);
  EXPECT_EQ(testHBuf[3], 4);
}

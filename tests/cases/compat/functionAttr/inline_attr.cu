#include "test_buf.h"

__noinline__ __device__ __host__ int func1() {
  return 1;
}

__attribute__((noinline)) __device__ __host__ int func2() {
  return 2;
}

__attribute__((__noinline__)) __device__ __host__ int func3() {
  return 3;
}

__attribute__((global)) void kernel_inline_attr(int* dp) {
  dp[0] = func1() + func2() + func3();
}

TEST(inline_attr, noinline) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel_inline_attr<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 6);
}
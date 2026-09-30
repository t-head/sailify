#include "test_buf.h"

__device__ int mod16(int num) {
  __builtin_assume(num > 0); 
  return num % 16;
}

__global__ void kernel(int *dp) {
  dp[0] = mod16(dp[0]);
}

TEST(compiler_hint, builtin_assume) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  testHBuf[0] = 31;
  testDBuf = testHBuf;
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 15);
}
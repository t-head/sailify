#include "test_buf.h"
__device__ int get(int input)
{
  switch (input)
  {
    case 1: return 4;
    case 2: return 10;
    default: __builtin_unreachable();return 5;
  }
}

__global__ void kernel(int *dp) {
  dp[0] = get(dp[0]);
}

TEST(compiler_hint, builtin_unreachable) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  testHBuf[0] = 2;
  testDBuf = testHBuf;
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 10);
}
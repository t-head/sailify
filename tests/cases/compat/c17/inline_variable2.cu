#include "test_buf.h"
inline __device__ int xxx = 10;

__global__ void kernel(int *dp) {
  dp[0] = xxx;
}

TEST(inline_variable, shared) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 10);
}
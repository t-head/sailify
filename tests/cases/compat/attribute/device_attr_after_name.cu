#include "test_buf.h"

__noinline__ int f __device__ (int *dp) {
  return dp[0] + dp[1];
}

__global__ void kernel(int *dp) {
  dp[0] = f(dp);
}

void kernel2 __global__(int *dp) {
  dp[0] = 10;
}

TEST(device_attr, attr_after_name) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  testHBuf.initialize([](int i) {return i + 1;});
  testDBuf = testHBuf;
  kernel<<<1, 1>>>(testDBuf);
  kernel2<<<1, 1>>>(testDBuf.get_pointer() + 1);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 3);
  EXPECT_EQ(testHBuf[1], 10);
}
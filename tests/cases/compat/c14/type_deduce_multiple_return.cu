#include "test_buf.h"

struct S {
  float v;
  __device__ S(int n) : v{2.0f * n} {}
  auto __device__ get_v() {
    return (int)v;
  }
};

__global__ void kernel(int *dp) {
  auto lam = [] (int x) {
    if (x < 0) {
        S ss(x);
        return ss.get_v();
    } else {
        return x + 1;
    }
  };
  dp[0] = lam(dp[0]);
  dp[1] = lam(dp[1]);
}

TEST(type_deduce, multiple_return) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  testHBuf[0] = 5; testHBuf[1] = -1;
  testDBuf = testHBuf;
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 6);
  EXPECT_EQ(testHBuf[1], -2);
}
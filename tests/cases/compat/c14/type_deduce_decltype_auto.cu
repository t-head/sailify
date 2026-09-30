#include "test_buf.h"

struct S {
  int v{0};
  __device__ auto incr_v() {
    v ++;
    return *this;
  }
  __device__ decltype(auto) incr_v2() {
    v ++;
    return *this;
  }
  __device__ void multi() {
    v *= 2;
  }
};

__global__ void kernel(int *dp) {
    S s1, s2;
    for (int i = 0; i < 3; i ++) {
        s1.incr_v().multi();
        s2.incr_v2().multi();

    }
    dp[0] = s1.v;
    dp[1] = s2.v;
    auto lam = [&]()->auto& { return s1.incr_v2(); };
    lam().multi();
    dp[2] = s1.v;
}

TEST(type_deduce, decltype_auto) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 3);
  EXPECT_EQ(testHBuf[1], 14);
  EXPECT_EQ(testHBuf[2], 8);
}
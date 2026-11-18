#include <utility>
#include "test_buf.h"

struct S {
  int val;
  __host__ __device__ S(int v) : val{v} {}
  __host__ __device__ S(S&& ss) {
    val = ss.val;
    ss.val = 0;
  }
  __host__ __device__ S(S& ss) {
    val = ss.val;
  }
  __host__ __device__ int get_v() {
    return val + 1;
  }
};

__global__ void kernel(int *dp) {
    S ss = S(dp[0]);
    S ss2 = std::move(ss);
    dp[0] = ss2.get_v();
    dp[1] = ss.get_v();
}

TEST(rvalue_reference, constructor_overloading) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  testHBuf.initialize([](int i) {return i + 10;});
  testDBuf = testHBuf;
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 11);
  EXPECT_EQ(testHBuf[1], 1);
}

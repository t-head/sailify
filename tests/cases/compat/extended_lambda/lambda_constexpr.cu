#include "test_buf.h"

template<typename LAM>
__host__ __device__ auto lam_in_lam(LAM lam, int v) {
  return [v](LAM lam) {
    return lam(v);
  }(lam);
}

__global__ void kernel_lam_in_lam(int *dp) {
  dp[0] = lam_in_lam(
    [](int v) { return v + 1;}, 100
  );
}

TEST(extended_lambda, lam_in_lam) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel_lam_in_lam<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 101);
}
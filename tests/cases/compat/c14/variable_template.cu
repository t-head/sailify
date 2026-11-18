#include "test_buf.h"
namespace aa {
  template<typename T>
    constexpr T __device__ pi = T(3.14159265358979323846);
};

__global__ void kernel(int *dp) {
    float x = aa::pi<float>;
    dp[0] = x;
}

TEST(variable_template, namespace) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 3);
}
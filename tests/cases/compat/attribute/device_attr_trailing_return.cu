#include "test_buf.h"

template<typename T>
auto func (T var) __device__ -> decltype(var * 2.0) {
  return var;
}

__global__ void kernel(double *dp) {
  auto a = func(1.0f);
  static_assert(std::is_same<decltype(a), double>::value, "expected double type, not match");
  dp[0] = a;
}

TEST(device_attr, trailing_return) {
  auto testDBuf = testBuf<double, memory_scope::device>(1);
  auto testHBuf = testBuf<double, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 1.0);
}
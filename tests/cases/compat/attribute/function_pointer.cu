#include "test_buf.h"

__device__ int add_v(int a, int b) {
  return a + b;
}

__device__ int sub_v(int a, int b) {
  return a - b;
}

__global__ void kernel(int*dp, int flag) {
  auto fp = (flag == 1) ? add_v : sub_v;
  dp[0] = fp(dp[1], dp[2]);
}

template<typename dfunc>
int get_flag(dfunc df) {
  return (df == &add_v) ? 1 : 2;
}

TEST(attribute, function_pointer) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  testHBuf.initialize([](int i) {return i;});
  testDBuf = testHBuf;
  auto fp = &add_v;
  auto dp = testDBuf.get_pointer();
  kernel<<<1, 1>>>(dp, get_flag(fp));
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 3);
}
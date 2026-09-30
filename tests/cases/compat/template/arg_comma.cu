#include "test_buf.h"

template<
  typename A,
  typename B,
  int      C
>
struct S;

template<
  typename A,
  typename B
>
struct S<
  A,
  B,
  2,
> {
  A v = 2;
};

__global__ void kernel(int *dp) {
  S<int, float, 2> ss;
  dp[0] = ss.v;
}

TEST(template, arg_comma) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  auto hval = testHBuf[0];
  TLOG_INFO << std::hex << "Loaded value: 0x" << hval << "\n";
  EXPECT_EQ(testHBuf[0], 2);
}
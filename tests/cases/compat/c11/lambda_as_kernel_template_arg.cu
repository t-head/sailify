#include "test_buf.h"

typedef int(*fp)();
__global__ void k1(fp* ptr) {
  auto l2 = [] { return 2; };
  ptr[0] = l2;
}

template<typename T>
__global__ void k2(T func, int *dp) {
  dp[0] = func();
}

TEST(lambda, kernel_template_arg) {
  auto testDFUNCBuf = testBuf<fp, memory_scope::device>(1);
  auto testHFUNCBuf = testBuf<fp, memory_scope::host>(1);
  k1<<<1, 1>>>(testDFUNCBuf);
  testHFUNCBuf = testDFUNCBuf;
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  k2<<<1, 1>>>(testHFUNCBuf[0], testDBuf.get_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 2);
}

template<typename T>
__global__ void kernel(int *dp, T lfunc) {
    dp[0] = lfunc(dp[0]);
}
void f(int *dp) {
  auto lam = [] __host__ __device__(int x) {return x * 10;};
  kernel<<<1, 1>>>(dp, lam);
}

TEST(lambda, extended_lambda) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  testHBuf[0] = 2;
  testDBuf = testHBuf;
  f(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 20);
}

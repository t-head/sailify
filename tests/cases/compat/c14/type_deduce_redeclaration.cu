#include "test_buf.h"
struct A {
  __device__ auto f();
};
auto A::f() { return 19; }

__global__ void kernel(int *dp) {
  A aa;
  dp[0] = aa.f();
}

TEST(type_deduce, redeclaration) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 19);
}

template<typename T>
__device__ auto f(T v);

template<>
__device__ auto f(int v) {
  return v + 1;
}
template<>
__device__ auto f(float v) {
    return v + 2;
}

__global__ void kernel2(int *dp) {
    dp[0] = f(10);
    dp[1] = f(20.0f);
}

TEST(type_deduce, template) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel2<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 11);
  EXPECT_EQ(testHBuf[1], 22);
}

__device__ auto sum(int i) {
  if (i == 1)
    return i * 1.0;
  else
    return sum(i-1)+i;
}

__global__ void kernel3(double *dp) {
  dp[0] = sum(3);
}

TEST(type_deduce, recursion) {
  auto testDBuf = testBuf<double, memory_scope::device>(1);
  auto testHBuf = testBuf<double, memory_scope::host>(1);
  kernel3<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_FLOAT_EQ(testHBuf[0], 6.0);
}

auto __host__ __device__ get_lam(int *x) {
  return [=]() {return (*x)++ + 1;};
}

__global__ void kernel4(int *dp) {
  dp[1] = get_lam(dp)();
}

TEST(type_deduce, lambda) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  testHBuf[0] = 5;
  testDBuf = testHBuf;
  kernel4<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 6);
  EXPECT_EQ(testHBuf[1], 6);
}
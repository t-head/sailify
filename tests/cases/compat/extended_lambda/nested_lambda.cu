#include "test_buf.h"

template<typename T>
__global__ void kernel(int *dp, T lam) {
  dp[0] = lam();
}

void foo(int *dp) {
  auto lam2 = [=]  {
     auto lam3 = [=] {
        auto lam4 = []  __host__ __device__ {
            return 10;
        };
        kernel<<<1, 1>>>(dp, lam4);
     };
     lam3();
  };
  lam2();
}

TEST(extended_lambda, nested_lambda) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  testDBuf.clear();
  foo(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 10);
}

__device__ int nested_implicit_capture() {
  int a = 1, b = 2, c = 3;
  auto lam = [=] __device__ () mutable {
    a += 10; b += 20; c += 30;
    return [=] __device__ () mutable {
      a += 100; b += 200; c += 300;
      return [=] __device__() mutable {
        return a + b + c;
      } () + 1;
    } () + 1;
  };
  a = 1000; b = 2000; c = 3000;
  return lam();
}

__global__ void kernel_nested_lambda_implicit_capture(int* dp) {
  dp[0] = nested_implicit_capture();
}

TEST(extended_lambda, nested_lambda_implicit_capture) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel_nested_lambda_implicit_capture<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 668);
}

__device__ int nested_ref_capture(int& a, int& b, int& c) {
  auto m1 = [a, &b, &c] () mutable {
    auto m2 = [a, b, &c] __device__ __host__ () mutable {
      auto v = a + b + c;
      a = 1000; b = 2000; c = 3000;
      return v;
    };
    a = 100; b = 200; c = 300;
    return m2();
  };
  a = 10; b = 20; c = 30;
  return m1();
}

__global__ void kernel_nested_ref_capture(int *dp) {
  int a = 1, b = 2, c = 3;
  dp[0] = nested_ref_capture(a, b, c);
  dp[1] = a + b + c;
}

TEST(extended_lambda, nested_lambda_ref_capture) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_nested_ref_capture<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 321);
  EXPECT_EQ(testHBuf[1], 3210);
}
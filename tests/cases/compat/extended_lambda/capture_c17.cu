#include "test_buf.h"

template<typename T1, typename T2>
__global__ void kernel(int *dp, T1 lam, T2 lam1) {
  dp[0] = lam();
  dp[1] = lam1();
}

void use_lam(int &hp, int *dp) {
  int yyy = hp + 1;
  auto lam = [yyy]__device__{
    if constexpr(true) {
      return yyy * 2;
    }
    return 1;};
  auto lam2 = [=]__device__{
    int result = yyy;
    if constexpr(true) {
      return result * 3;
    }
    return 1;};
  kernel<<<1, 1>>>(dp, lam, lam2);
}

TEST(extended_lambda, capture_c17) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  testDBuf.clear();
  int h = 10;
  use_lam(h, testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 22);
  EXPECT_EQ(testHBuf[1], 33);
}

struct capture_star_this {
  int v;
  __device__ capture_star_this(int v) : v{v} {}
  __device__ int copy() {
    return [=, *this] __device__ __host__ () mutable {return ++v;}();
  }
};

__global__ void kernel_capture_star_this(int* dp) {
  auto ss = capture_star_this(100);
  dp[0] = ss.copy();
  dp[1] = ss.v;
}

TEST(extended_lambda, capture_lambda_star_this) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_capture_star_this<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 101);
  EXPECT_EQ(testHBuf[1], 100);
}

struct capture_nested
{
  int v = 10;
  __device__ auto operator()() 
  {
      return [this]
      {
          return [*this] () mutable
          {
              return ++v;
          }();
      }();
  }
};

__global__ void kernel_capture_nested(int* dp) {
  capture_nested cns;
  dp[0] = cns();
  dp[1] = cns.v;
}

TEST(extended_lambda, capture_lambda_nested) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_capture_nested<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 11);
  EXPECT_EQ(testHBuf[1], 10);
}
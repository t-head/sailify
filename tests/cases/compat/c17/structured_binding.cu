#include "test_buf.h"

__device__ int arr[] = {1, 2, 3};

__global__ void kernel(int *dp) {
    auto [a1, a2, a3] = arr;
    dp[0] = a1;
    dp[1] = a2;
    dp[2] = a3;
}

TEST(structured_binding, array) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 1);
  EXPECT_EQ(testHBuf[1], 2);
  EXPECT_EQ(testHBuf[2], 3);
}

struct S {
  int n;
  float f;
  double d;
};

__device__ S ss{1, 2.0f, 3.0};

__global__ void k2(int *dp) {
  auto &[a, b, c] = ss;
  a ++; b ++; ++ c;
  dp[0] = ss.n;
  dp[1] = ss.f;
  dp[2] = ss.d;
}

TEST(structured_binding, struct) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  k2<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 2);
  EXPECT_EQ(testHBuf[1], 3);
  EXPECT_EQ(testHBuf[2], 4);
}

struct bitfield {
  unsigned a : 4;
  unsigned b : 8;
  unsigned c : 20;
};

__global__ void k3(unsigned *dp) {
    bitfield ss;
    auto &[a, b, c] = ss;
    a = 0xA;
    b = 0xBB;
    c = 0xCCCCC;
    dp[0] = *(unsigned*)&ss;
}

TEST(structured_binding, bitfield) {
  auto testDBuf = testBuf<unsigned, memory_scope::device>(1);
  auto testHBuf = testBuf<unsigned, memory_scope::host>(1);
  k3<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 0xcccccbba);
}
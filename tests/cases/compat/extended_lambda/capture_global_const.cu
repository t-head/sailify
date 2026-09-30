#include "test_buf.h"

__device__ int gvar = 10, gvar2 = 20;

__global__ void kernel(int* dp) {
  [&, dp]{
    atomicAdd(dp, gvar2 + atomicAdd(&gvar, 1));
  }();
}

TEST(extended_lambda, capture_global) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  auto dp = testDBuf.get_initialized_pointer();
  kernel<<<8, 1>>>(dp);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 268);
}

__global__ void kernel2(int* dp) {
  const const int a = 10;
  constexpr int b = 20;
  const int const c = 30;
  [](int* p){
    p[threadIdx.x] += a + b + c;
  }(dp);
}

TEST(extended_lambda, capture_const) {
  auto testDBuf = testBuf<int, memory_scope::device>(4);
  auto testHBuf = testBuf<int, memory_scope::host>(4);
  auto dp = testDBuf.get_initialized_pointer();
  kernel2<<<1, 4>>>(dp);
  testHBuf = testDBuf;
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = 60;
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running result: " << (pass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(pass);
}

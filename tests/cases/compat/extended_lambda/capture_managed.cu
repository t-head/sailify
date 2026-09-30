#include "test_buf.h"

__managed__ int mvar[2][2][3], mvar2[2][6];

__global__ void kernel3(int* dp) {
  int tid = threadIdx.x;
  int idx3 = tid % 3;
  int idx2 = (tid / 3) % 2;
  int idx1 = (tid / 6) % 2;
  int idx5 = tid % 6;
  int idx4 = (tid / 6) % 2;
  [&] (int *p) -> void{
    dp[tid] = ++mvar[idx1][idx2][idx3];
    dp[tid] += mvar2[idx4][idx5]++;
  }(dp);
}

TEST(extended_lambda, capture_managed) {
  auto testDBuf = testBuf<int, memory_scope::device>(12);
  auto testHBuf = testBuf<int, memory_scope::host>(12);
  auto dp = testDBuf.get_initialized_pointer();

  for (int* p = &mvar[0][0][0], *q = &mvar2[0][0], l = 0; l < 12; l ++) {
    *(p++) = 1;
    *(q++) = 5;
  }
  kernel3<<<1, 12>>>(dp);
  testHBuf = testDBuf;
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = 7;
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Checking kernel result: " << (pass ? "pass" : "fail") << "\n";
  for (int* p = &mvar[0][0][0], *q = &mvar2[0][0], l = 0; l < 12; l ++) {
    auto v1 = *(p++);
    auto v2 = *(q++);
    bool check1 = (v1 == 2);
    bool check2 = (v2 == 6);
    pass &= check1 & check2;
    if (!check1 || !check2) {
      TLOG_ERROR << "l: " << l << ", v1: " << v1 << ", v2: " << v2 << "\n";
    }
  }
  EXPECT_TRUE(pass);
}
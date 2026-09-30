#include "test_buf.h"

__global__ void kernel1(int *dp) {
  const int &k1 = threadIdx.x;
  const int &k2 = threadIdx.y;
  const int &k3 = threadIdx.z;
  const int &k4 = blockIdx.x;
  const int &k5 = blockIdx.y;
  const int &k6 = blockIdx.z;
  const int &k7 = gridDim.x;
  const int &k8 = gridDim.y;
  const int &k9 = gridDim.z;
  const int &k10 = blockDim.x;
  const int &k11 = blockDim.y;
  const float k12 = blockDim.z;

  if (k1 == 0 && k2 == 0 && k3 == 9 && k4 == 0 && k5 == 0 && k6 == 9) {
    dp[0] = k1;
    dp[1] = k2;
    dp[2] = k3;
    dp[3] = k4;
    dp[4] = k5;
    dp[5] = k6;
    dp[6] = k7;
    dp[7] = k8;
    dp[8] = k9;
    dp[9] = k10;
    dp[10] = k11;
    dp[11] = k12;
  }
}

TEST(regression, builtin_dim3_var_ref) {
  auto testDBuf = testBuf<int, memory_scope::device>(12);
  auto testHBuf = testBuf<int, memory_scope::host>(12);
  dim3 blocks(2, 3, 10);
  dim3 tpb(5, 2, 10);
  kernel1<<<blocks, tpb>>>(testDBuf);
  testHBuf = testDBuf;
  
  int goldens[] = {0, 0, 9, 0, 0, 9, 2, 3, 10, 5, 2, 10};
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    bool check = (v == goldens[i]);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Index: " << i << ", EXPECT: " << goldens[i]
        << ", ACTUAL: " << v << "\n";
    }
  });
  EXPECT_TRUE(pass);
}

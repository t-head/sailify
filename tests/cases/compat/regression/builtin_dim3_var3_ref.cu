#include "test_buf.h"

__managed__ int dp[12];

__global__ void kernel() {
  auto lam = [&, v1=threadIdx, v2=blockIdx,
    v3=gridDim, v4=blockDim.x, v5=blockDim.y,
    v6 = blockDim.z]() {
    if (v1.x == 0 && v1.y == 1 && v1.z == 0 && v2.x == 1
      && v2.y == 2 && v2.z == 3) {
      int arr[] = {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z,
        v4, v5, v6};
      auto [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] = arr;
      dp[0] = a1;
      dp[1] = a2;
      dp[2] = a3;
      dp[3] = a4;
      dp[4] = a5;
      dp[5] = a6;
      dp[6] = a7;
      dp[7] = a8;
      dp[8] = a9;
      dp[9] = a10;
      dp[10] = a11;
      dp[11] = a12;
    }
  };
  lam();
}

TEST(regression, builtin_dim3_var3_ref) {
  auto testHBuf = testBuf<int, memory_scope::host>(12);
  dim3 blocks(2, 3, 10);
  dim3 tpb(5, 3, 10);
  kernel<<<blocks, tpb>>>();
  cudaDeviceSynchronize();
  
  int goldens[] = {0, 1, 0, 1, 2, 3, 2, 3, 10, 5, 3, 10};
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    bool check = (dp[i] == goldens[i]);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Index: " << i << ", EXPECT: " << goldens[i]
        << ", ACTUAL: " << dp[i] << "\n";
    }
  });
  EXPECT_TRUE(pass);
}
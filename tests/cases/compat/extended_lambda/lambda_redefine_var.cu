#include "test_buf.h"

__managed__ int dp[12];

__global__ void kernel() {
  auto lam = [x=10]() {
    int x = blockIdx.x;
    dp[threadIdx.x + x * blockDim.x] = x;
  };
  lam();
}

TEST(extended_lambda, redefine_var) {
  auto testHBuf = testBuf<int, memory_scope::host>(12);
  dim3 blocks(12, 1, 1);
  dim3 tpb(1, 1, 1);
  kernel<<<blocks, tpb>>>();
  cudaDeviceSynchronize();
  
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    bool check = (dp[i] == i);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Index: " << i << ", EXPECT: " << i
        << ", ACTUAL: " << dp[i] << "\n";
    }
  });
  EXPECT_TRUE(pass);
}
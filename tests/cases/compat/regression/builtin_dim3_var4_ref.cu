#include "test_buf.h"

__managed__ unsigned dp[12];

struct S {
  uint3 v1{blockIdx};
  uint3 v2{threadIdx};
  dim3 v3{blockDim};
  unsigned v4{gridDim.x};
  unsigned v5{gridDim.y};
  unsigned v6{gridDim.z};
};

__global__ void kernel() {
  S ss;
  if (ss.v1.x == 1 && ss.v1.y == 0 &&
      ss.v1.z == 2 && ss.v2.x == 2 &&
      ss.v2.y == 1 && ss.v2.z == 5) {
    dp[0] = ss.v1.x;
    dp[1] = ss.v1.y;
    dp[2] = ss.v1.z;
    dp[3] = ss.v2.x;
    dp[4] = ss.v2.y;
    dp[5] = ss.v2.z;
    dp[6] = ss.v3.x;
    dp[7] = ss.v3.y;
    dp[8] = ss.v3.z;
    dp[9] = ss.v4;
    dp[10] = ss.v5;
    dp[11] = ss.v6;
  }
}

TEST(regression, builtin_dim3_var4_ref) {
  auto testHBuf = testBuf<int, memory_scope::host>(12);
  dim3 blocks(2, 3, 10);
  dim3 tpb(5, 3, 10);
  kernel<<<blocks, tpb>>>();
  cudaDeviceSynchronize();
  
  int goldens[] = {1, 0, 2, 2, 1, 5, 5, 3, 10, 2, 3, 10};
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
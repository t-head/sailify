#include "test_buf.h"

inline namespace N1 {
  namespace N2 {
    __device__ int Gvar = 2;
  }
  namespace N3 {
    __global__ void k1(int *dp) {
        dp[0] = N2::Gvar + 1;
    }
  }
  namespace N4 {
    template<typename T>
    __device__ T f() {
        return static_cast<T>(N2::Gvar + 2);
    }
  }
}

__global__ void kernel(int *dp) {
    dp[1] = N2::Gvar;
    dp[2] = N4::f<int>();
}

TEST(inline_namespace, test) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  N3::k1<<<1,1>>>(testDBuf);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 3);
  EXPECT_EQ(testHBuf[1], 2);
  EXPECT_EQ(testHBuf[2], 4);
}

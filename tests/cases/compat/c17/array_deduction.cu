#include <array>
#include "test_buf.h"

enum Color {BLUE=0, RED=10, YELLOW=20};

__device__ auto get_array() {
  constexpr std::array arr = {BLUE, RED, YELLOW};
  return arr;
}

__global__ void kernel(int *dp) {
  std::array a = get_array();
  dp[0]= a[0] + a[1] + a[2];
}

TEST(c17, array_deduction) {
  auto testDbuf = testBuf<int, memory_scope::device>(1);
  auto testHbuf = testBuf<int, memory_scope::host>(1);
  auto dp = testDbuf.get_initialized_pointer();
  kernel<<<1, 1>>>(dp);
  testHbuf = testDbuf;
  EXPECT_EQ(testHbuf[0], 30);
}

#include "test_buf.h"

#define HD __host__ __device__
class S {
public:
  mutable int v{0};
  HD int  get() const { return v; }
  HD void inc() const { v++; }
};

__global__ void kernel(int *dp) {
  const S ss{};
  ss.v = 11;
  dp[0] = ss.get();
  ss.inc();
  dp[1] = ss.get();
}

TEST(c11, mutable_keyword) {
  auto testDbuf = testBuf<int, memory_scope::device>(2);
  auto testHbuf = testBuf<int, memory_scope::host>(2);
  auto dp = testDbuf.get_initialized_pointer();
  kernel<<<1, 1>>>(dp);
  testHbuf = testDbuf;
  EXPECT_EQ(testHbuf[0], 11);
  EXPECT_EQ(testHbuf[1], 12);
}
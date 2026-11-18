#include "test_buf.h"

__global__ void kernel(uint32_t *dp) {
  float f1 = *reinterpret_cast<float*>(dp + 1);
  float f2 = *reinterpret_cast<float*>(dp + 2);
  float f0 = f1 + f2;
  dp[0] = *reinterpret_cast<uint32_t*>(&f0);
}

TEST(COMPILER_HINT, OPTION_FTZ) {
  auto testDbuf = testBuf<uint32_t, memory_scope::device>(3);
  auto testHbuf = testBuf<uint32_t, memory_scope::host>(3);
  auto dp = testDbuf.get_pointer();
  uint32_t hf[] = {0x3, 0x1, 0x1};
  testHbuf.initialize([&](int i){return hf[i];});
  testDbuf = testHbuf;
  kernel<<<1, 1>>>(dp);
  testHbuf = testDbuf;
  EXPECT_EQ(testHbuf[0], 0);
}
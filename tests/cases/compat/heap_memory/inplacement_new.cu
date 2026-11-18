#include "test_buf.h"

__global__ void kernel(char *dp) {
  char buf[12] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0};
  int *hp = new((int*)buf) int;
  hp[0] = 0x04030201; hp[1] = 0x08070605; hp[2] = 0x0c0b0a09;
  memcpy(dp, buf, sizeof(char) * 12);
}

TEST(heap_memory, inplacement_new) {
  auto testDBuf = testBuf<char, memory_scope::device>(12);
  auto testHBuf = testBuf<char, memory_scope::host>(12);
  testDBuf.clear();
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  testHBuf.foreach([&](char v, int i) {
    EXPECT_EQ(v, i + 1);
  });
}
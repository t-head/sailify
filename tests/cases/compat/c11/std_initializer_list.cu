#include <initializer_list>
#include "test_buf.h"

__device__ int foo(std::initializer_list<int> in) {
    int ret = 0;
    for (auto x: in) {
        ret += x;
    }
    return ret;
};

__global__ void kernel(int *dp)
{
    dp[0] = foo({4,5,6});

    int i = 10;
    dp[1] = foo({i,5,6});
}

TEST(initializer_list, inside_device) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 15);
  EXPECT_EQ(testHBuf[1], 21);
}

#include <utility>
#include "test_buf.h"

__device__ int f(int&& val) {
    return 2;
}
__device__ int f(int& val) {
    return 1;
}
template<typename T>
__device__ int forwarder(T&& val) {
    return f(std::forward<T>(val));
}

__global__ void kernel(int *dp) {
    dp[0] = forwarder(1);
    int x = 10;
    dp[1] = forwarder(x);
}
TEST(rvalue_reference, std_forward) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 2);
  EXPECT_EQ(testHBuf[1], 1);
}

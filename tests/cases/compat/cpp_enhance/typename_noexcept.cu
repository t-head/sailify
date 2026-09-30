#include "test_buf.h"

template<typename T>
struct S {
  struct inside {
    __device__ inside(T x) : v{x} {}
    T v;
  };
};

template<typename T>
__device__ void f(int *dp) {
  auto uu = noexcept(std::declval<S<T>::inside>());
  dp[0] = uu;
}

__global__ void kernel(int *dp) {
  f<int>(dp);
}

TEST(cpp_enhance, typename_noexcept) {
  auto testDInput = testBuf<int, memory_scope::device>(1);
  auto testHOut = testBuf<int, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDInput);
  testHOut = testDInput;
  EXPECT_EQ(testHOut[0], 1);
}
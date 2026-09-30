#include "test_buf.h"

class A {
public:
    int work(int *p) {
    int arr[count_] = {0};
    memcpy(arr, p, sizeof(int) * count_);
    return arr[0] + arr[count_ - 1];
  }
private:
  const int count_ = 10;
};

__host__ void kernel(int *dp) {
  A a;
  dp[0] = a.work(dp);
}

TEST(class, friend_member_function) {
  auto testHBuf = testBuf<int, memory_scope::host>(10);
  testHBuf = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  kernel(testHBuf);
  EXPECT_EQ(testHBuf[0], 11);
}
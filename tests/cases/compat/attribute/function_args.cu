#include "test_buf.h"

void f(__device__ int& var) {
  var = 1;
}

TEST(func_arg, dev_attr) {
  int v = 2;
  f(v);
  EXPECT_EQ(v, 1);
}
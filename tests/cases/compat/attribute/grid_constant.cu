#include "test_buf.h"

struct S {
    int x;
    int y;
    int z;
    int t;
    S() {
        x = 10;
        y = 5;
        z = 0;
        t = 0;
    }
};

__device__ __noinline__ int device_func(const S &s1);
__global__ void kernel_func(const __grid_constant__ S s1, int *dev_data);

TEST(attribute, grid_constant) {
  unsigned int num = 32*1024;
  S s1;
  auto testDBuf = testBuf<int, memory_scope::device>(num);
  auto testHBuf = testBuf<int, memory_scope::host>(num);
  testHBuf.initialize([](int i) {return 10;});
  testDBuf = testHBuf;
  auto dp = testDBuf.get_pointer();
  kernel_func<<<32, 1024>>>(s1, dp);
  testHBuf = testDBuf;
  int ret = 1;
  for(int i = 0; i < num; i++){
    if (testHBuf[i] != 85){
        ret = 0;
        printf("idx: %d, expect: 85, actual: %d",i,testHBuf[i]);
        break;
    }
  }
  EXPECT_EQ(ret, 1);
}

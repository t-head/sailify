#include "test_buf.h"

__global__ void kernel_func(const int __grid_constant__ s1, int *dev_data) {
    void* constantPtr = __cvta_grid_constant_to_generic((size_t)s1);
    size_t constantValue = __cvta_generic_to_grid_constant(constantPtr);
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    dev_data[tid] += constantValue;
}


TEST(attribute, cvta_grid_constant) {
  unsigned int num = 32*1024;
  const int s1 = 10;
  auto testDBuf = testBuf<int, memory_scope::device>(num);
  auto testHBuf = testBuf<int, memory_scope::host>(num);
  testHBuf.initialize([](int i) {return 10;});
  testDBuf = testHBuf;
  auto dp = testDBuf.get_pointer();
  kernel_func<<<32, 1024>>>(s1, dp);
  testHBuf = testDBuf;
  int ret = 1;
  for(int i = 0; i < num; i++){
    if (testHBuf[i] != 20){
        ret = 0;
        printf("idx: %d, expect: 20, actual: %d",i,testHBuf[i]);
        break;
    }
  }
  EXPECT_EQ(ret, 1);
}



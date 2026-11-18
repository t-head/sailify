#include "test_buf.h"

__device__ int sum(int const& s1){
    if( __isGridConstant((void*)&s1)){
        return s1;
    }
    return 0;
}

__global__ void kernel_func(const __grid_constant__ int s1, int *dev_data) {
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    dev_data[tid] += sum(s1);
}


TEST(attribute, is_grid_constant) {
  unsigned int num = 32*1024;
  const int s1=10;
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
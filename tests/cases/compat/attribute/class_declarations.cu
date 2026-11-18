#include "test_buf.h"

__global__ void kernel() {
  __shared__ class A;
  __device__ class B;
  __constant__ class C;
  __managed__ class D;
}

__device__ void kernel2() {
  __shared__ class A;
  __device__ class B;
  __constant__ class C;
  __managed__ class D;
}

TEST(class_declaration, device_attr) {
  EXPECT_TRUE(1);
}
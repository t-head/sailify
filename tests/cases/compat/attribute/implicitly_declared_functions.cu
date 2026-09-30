#include "test_buf.h"

class Base {
  int x;
public:
  __host__ __device__ Base(void) : x(10) {}
  __host__ __device__ int get_x() {return x;}
};

class Derived : public Base {
  int y;
};

class Other: public Base {
  int z;
};

__device__ void foo(int *dp)
{
  Derived D1;
  Other D2;
  dp[0] = D1.get_x() + D2.get_x();
}

__host__ void bar(int *hp)
{
  Other D3;
  hp[0] = D3.get_x();
}

__global__ void kernel_attr_implicit(int *dp) {
  foo(dp);
}

TEST(attribute, implicitly_declared_function) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_attr_implicit<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  bar(testHBuf.get_pointer() + 1);
  EXPECT_EQ(testHBuf[0], 20);
  EXPECT_EQ(testHBuf[1], 10);
}

struct Base1 { virtual __host__ __device__ ~Base1() { } };
struct Derived1 : Base1 { };

struct Base2 { virtual __device__ ~Base2(); };
__device__ Base2::~Base2() = default;
struct Derived2 : Base2 { };

__global__ void kernel_attr_destructor() {
    Derived1 d1;
}

Derived2* f(Derived2 &d) {
  return &d;
}

TEST(attribute, implicitly_declared_destructor) {
  Derived2 d2;
  auto *ptr = f(d2);
  EXPECT_EQ(ptr, &d2);
  auto ptr2 = new Derived2;
  delete ptr2;
}
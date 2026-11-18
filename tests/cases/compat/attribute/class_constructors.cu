#include "test_buf.h"

class A {
public:
  A() {};
};

__device__ A b;
__constant__ A c;
__managed__ A d;

class B {
public:
  __device__ B() {};
private:
  int v;
};

__device__ B bb;
__constant__ B cv;
__managed__ B dd;

__device__ void kernel() {
  __shared__ A a;
  __shared__ B b;
}

__global__ void kernel2() {
  __shared__ A a;
  __shared__ B b;
}

class C {
public:
  __device__ __host__ C(int a) : v{a} {};
  C() {return;};
private:
  int v;
};

__device__ C b2;
__constant__ C c2;
__managed__ C d2;

__device__ void kernel3() {
  __shared__ C a;
}

__global__ void kernel4() {
  __shared__ C a;
}

class base {
public:
  virtual void f() = 0;
};

class D : public virtual base {
public:
  D() = default;
  virtual void f();
  virtual int get() {return v;}
private:
  int v{100};
};

__device__ void kernel5() {
  __shared__ D a;
}

__global__ void kernel6() {
  __shared__ D a;
}

class base2 {
protected:
   virtual void f() = 0;
   base2() {fv = 1.0f;}
private:
   float fv;
};
class AA : public virtual base2 {
public:
   __device__ AA() =default;
   virtual void f();
   virtual int get() {return v;}
private:
  int v{100};
};

__device__ void kernel7() {
  __shared__ AA a;
}

__global__ void kernel8() {
  __shared__ AA a;
}

class BBB {
public:
  BBB() {fv = 1.0f;}
private:
   float fv;
};
class AAA {
public:
  AAA() = default;
private:
  int v;
  BBB b;
};

__device__ void kernel9() {
  __shared__ AAA a;
}

__global__ void kernel10() {
  __shared__ AAA a;
}

TEST(class_constructor, device_attr) {
  EXPECT_TRUE(1);
}
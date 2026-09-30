#include "test_buf.h"

class A {
public:
  A() =default;
  ~A() {};
};

__device__ A b;
__constant__ A c;
__managed__ A d;

class B {
public:
  B() =default;
  ~B() {};
  virtual void f();
};

class C {
public:
   C() =default;
   ~C() {
    if (ptr) {
        free(ptr);
    }
  }
private:
  int* ptr{nullptr};
};

class D {
protected:
  __device__ void f() { return;}
  virtual void g(){};
  virtual ~D() {};
};

class E : public virtual D {
public:
    E() =default;
    ~E() {};
private:
  int* ptr{nullptr};
};

class F {
protected:
  __device__ void f() { return;}
  virtual void g(){};
   ~F() {
    if (ptr) {
      free(ptr);
    }
  };
private:
int* ptr{nullptr};
};

class G : public F {
public:
private:
int* ptr{nullptr};
};

class H {
public:
  __device__ void f() { return;}
  virtual void g(){};
  __device__ ~H() {
    if (ptr) {
      free(ptr);
    }
  }
private:
int* ptr{nullptr};
};

class I {
public:
    __device__ __host__ I() {};
    __device__ __host__ ~I() {};

private:
int* ptr{nullptr};
H b;
};

__device__ void kernel() {
  __shared__ A a;
  __shared__ B b;
  __shared__ C c;
  __shared__ E e;
  __shared__ G g;
  __shared__ I i;
}

__global__ void kernel2() {
  __shared__ A a;
  __shared__ B b;
  __shared__ C c;
  __shared__ E e;
  __shared__ G g;
  __shared__ I i;
}

extern __device__ I i2;
extern __constant__ I i3;
extern __managed__ I i4;

TEST(class_destructor, device_attr) {
  EXPECT_TRUE(1);
}
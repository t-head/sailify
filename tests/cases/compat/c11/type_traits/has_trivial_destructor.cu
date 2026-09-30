#define V(x) static_assert(x, "");
#include <cuda_fp16.h>
#include <cuda_bf16.h>
class A {};
class B {
  ~B() {}
};
class B2 {
  ~B2()=default;
};
class B3 {
  ~B3()=delete;
};
class B4 {
protected:
  virtual void g()=0;
  virtual ~B4(){};
};
class B5 {
  int v{1};
  int *p{nullptr};
  B5() {}
  B5(const B5& b) {
    v = b.v;
    p = b.p;
  }
  B5& operator=(const B5& b) {
    v = b.v;
    p = b.p;
    return *this;
  }
};
class B6 : protected B4 {
  ~B6() override =default;
};

__global__ void kernel() {
  V(__has_trivial_destructor(A))
  V(!__has_trivial_destructor(B))
  V(__has_trivial_destructor(B&))
  V(__has_trivial_destructor(B2))
  V(__has_trivial_destructor(B3))
  V(!__has_trivial_destructor(B4))
  V(__has_trivial_destructor(B5))
  V(!__has_trivial_destructor(B6))
  V(__has_trivial_destructor(__half))
  V(__has_trivial_destructor(__half2))
  V(__has_trivial_destructor(__nv_bfloat16))
  V(__has_trivial_destructor(__nv_bfloat162))
}

void f() {
  V(__has_trivial_destructor(A))
  V(!__has_trivial_destructor(B))
  V(__has_trivial_destructor(B&))
  V(__has_trivial_destructor(B2))
  V(__has_trivial_destructor(B3))
  V(!__has_trivial_destructor(B4))
  V(__has_trivial_destructor(B5))
  V(!__has_trivial_destructor(B6))
  V(__has_trivial_destructor(__half))
  V(__has_trivial_destructor(__half2))
  V(__has_trivial_destructor(__nv_bfloat16))
  V(__has_trivial_destructor(__nv_bfloat162))
}
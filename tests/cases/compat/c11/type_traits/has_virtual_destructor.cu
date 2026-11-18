#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");

union U {};
class C {};
class C2 {
  virtual void f() {}
};
class C3 {
  virtual void f(int)=0;
};
class C4 {
  virtual ~C4() {}
};
class C5 : public C2 {};
class C6 : virtual public C {};
class D {
public:
  virtual ~D()=0;
};
class D2 {
  virtual ~D2()=delete;
};
class D3 : public D {};
class D4 : public D {
  int *p;
public:
  ~D4() {
    if (p) free(p);
  }
};

__global__ void kernel() {
  V(!__has_virtual_destructor(U))
  V(!__has_virtual_destructor(C))
  V(!__has_virtual_destructor(C2))
  V(!__has_virtual_destructor(C3))
  V(__has_virtual_destructor(C4))
  V(!__has_virtual_destructor(C5))
  V(!__has_virtual_destructor(C6))
  V(!__has_virtual_destructor(__half))
  V(!__has_virtual_destructor(__half2))
  V(!__has_virtual_destructor(__nv_bfloat16))
  V(!__has_virtual_destructor(__nv_bfloat162))
  V(__has_virtual_destructor(D))
  V(__has_virtual_destructor(D2))
  V(__has_virtual_destructor(D3))
  V(__has_virtual_destructor(D4))
}

void f() {
  V(!__has_virtual_destructor(U))
  V(!__has_virtual_destructor(C))
  V(!__has_virtual_destructor(C2))
  V(!__has_virtual_destructor(C3))
  V(__has_virtual_destructor(C4))
  V(!__has_virtual_destructor(C5))
  V(!__has_virtual_destructor(C6))
  V(!__has_virtual_destructor(__half))
  V(!__has_virtual_destructor(__half2))
  V(!__has_virtual_destructor(__nv_bfloat16))
  V(!__has_virtual_destructor(__nv_bfloat162))
  V(__has_virtual_destructor(D))
  V(__has_virtual_destructor(D2))
  V(__has_virtual_destructor(D3))
  V(__has_virtual_destructor(D4))
}
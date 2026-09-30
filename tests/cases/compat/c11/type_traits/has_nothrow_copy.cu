#include<cuda_fp16.h>
#include<cuda_bf16.h>
class A {
public:
  int v;
};
class A2 {
public:
  A a;
};
class B {
public:
  int v;
  B(const B& b)noexcept:v{b.v}{} 
};
class B2 {
  int v;
  B2(B2&& b) throw() {v=b.v; b.v=0;}
};
class C: public A {
  int v{0};
  __device__ C(C& c) {v = c.v+1;}
public:
  C() {v=0;}
};
class C2 {
public:
  int v;
  C c;
  C2(const C2& c) noexcept {
    v = c.v;
  }
};
class D {
public:
  D(const D& d)=delete;
};
class D2 {
  D2(const D2& d)=default;
};
class D3 {
public:
  ~D3() {}
};
#define V(x) static_assert(x, "");

__global__ void kernel() {
  V(__has_nothrow_copy(A))
  V(__has_nothrow_copy(A2))
  V(__has_nothrow_copy(B))
  V(__has_nothrow_copy(B2))
  V(!__has_nothrow_copy(C))
  V(__has_nothrow_copy(C2))
  V(__has_nothrow_copy(C&))
  V(__has_nothrow_copy(D))
  V(__has_nothrow_copy(D2))
  V(__has_nothrow_copy(D3))
  V(__has_nothrow_copy(__half))
  V(!__has_nothrow_copy(__half2))
  V(__has_nothrow_copy(__nv_bfloat16))
  V(!__has_nothrow_copy(__nv_bfloat162))
}

void f() {
  V(__has_nothrow_copy(A))
  V(__has_nothrow_copy(A2))
  V(__has_nothrow_copy(B))
  V(__has_nothrow_copy(B2))
  V(!__has_nothrow_copy(C))
  V(__has_nothrow_copy(C2))
  V(__has_nothrow_copy(C&))
  V(__has_nothrow_copy(D))
  V(__has_nothrow_copy(D2))
  V(__has_nothrow_copy(D3))
  V(__has_nothrow_copy(__half))
  V(!__has_nothrow_copy(__half2))
  V(__has_nothrow_copy(__nv_bfloat16))
  V(!__has_nothrow_copy(__nv_bfloat162))
}
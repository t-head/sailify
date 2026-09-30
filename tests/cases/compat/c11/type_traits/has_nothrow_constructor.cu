#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>
#define V(x) static_assert(x, "");
class A {
};
class B : private A {
  V(__has_nothrow_constructor(A))
};
class C1 {
  __device__ C1()=default;
};
class C2 {
  C2()=default;
};
class D2 {
  D2() {}
};
class D3 {
  __attribute__((nothrow)) __host__ __device__ D3() {}
};
class D4 {
public:
  D4() noexcept {}
  D4(int) throw() {}
  D4(float) {}
};
class E {
  E()=default;
  D2 d;
};
class F {
  int *p{nullptr};
  F()=default;
};
class F2 {
public:
  int v {2};
};
class G {
  int v;
public:
  G(const G& g){v = g.v;}
};
class H {
  int v;
  H(const H&& h)=delete;
};
class K {
  __device__ K(){assert(1);}
};
const int x = 2;
class K2 {
  int v;
public:
  template<int N>
  K2() {v=N;}
};

__global__ void kernel() {
  V(__has_nothrow_constructor(A))
  V(__has_nothrow_constructor(B))
  V(__has_nothrow_constructor(C1))
  V(__has_nothrow_constructor(C2))
  class D {
    D(int x){}
    D()=delete;
  };
  V(__has_nothrow_constructor(D))
  V(!__has_nothrow_constructor(D2))
  V(__has_nothrow_constructor(D3))
  V(__has_nothrow_constructor(D4))
  V(!__has_nothrow_constructor(E))
  V(__has_nothrow_constructor(F))
  V(__has_nothrow_constructor(F2))
  V(!__has_nothrow_constructor(G))
  V(!__has_nothrow_constructor(H))
  V(__has_nothrow_constructor(__half))
  V(__has_nothrow_constructor(__half2))
  V(__has_nothrow_constructor(__nv_bfloat16))
  V(__has_nothrow_constructor(__nv_bfloat162))
  V(!__has_nothrow_constructor(K))
  V(!__has_nothrow_constructor(K2))
}

void f() {
  V(__has_nothrow_constructor(A))
  V(__has_nothrow_constructor(B))
  V(__has_nothrow_constructor(C1))
  V(__has_nothrow_constructor(C2))
  V(!__has_nothrow_constructor(D2))
  V(__has_nothrow_constructor(D4))
  V(!__has_nothrow_constructor(E))
  V(__has_nothrow_constructor(F))
  V(__has_nothrow_constructor(F2))
  V(!__has_nothrow_constructor(G))
  V(!__has_nothrow_constructor(H))
  V(__has_nothrow_constructor(__half))
  V(__has_nothrow_constructor(__half2))
  V(__has_nothrow_constructor(__nv_bfloat16))
  V(__has_nothrow_constructor(__nv_bfloat162))
  V(!__has_nothrow_constructor(K))
  V(!__has_nothrow_constructor(K2))
}
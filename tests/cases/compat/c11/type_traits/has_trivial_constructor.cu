#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define V(x) static_assert(x, "");
class A {
};
class B : private A {
  V(__has_trivial_constructor(A))
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
class E {
  E()=default;
  D2 d;
};
class F {
  int *p{nullptr};
  F()=default;
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

__global__ void kernel() {
  V(__has_trivial_constructor(A))
  V(__has_trivial_constructor(B))
  V(__has_trivial_constructor(C1))
  V(__has_trivial_constructor(C2))
  class D {
    D(int x){}
    D()=delete;
  };
  V(__has_trivial_constructor(D))
  V(!__has_trivial_constructor(D2))
  V(!__has_trivial_constructor(E))
  V(!__has_trivial_constructor(F))
  V(!__has_trivial_constructor(G))
  V(!__has_trivial_constructor(H))
  V(__has_trivial_constructor(__half))
  V(__has_trivial_constructor(__half2))
  V(__has_trivial_constructor(__nv_bfloat16))
  V(__has_trivial_constructor(__nv_bfloat162))
}

void f() {
  V(__has_trivial_constructor(A))
  V(__has_trivial_constructor(B))
  V(__has_trivial_constructor(C1))
  V(__has_trivial_constructor(C2))
  V(!__has_trivial_constructor(D2))
  V(!__has_trivial_constructor(E))
  V(!__has_trivial_constructor(F))
  V(!__has_trivial_constructor(G))
  V(!__has_trivial_constructor(H))
  V(__has_trivial_constructor(__half))
  V(__has_trivial_constructor(__half2))
  V(__has_trivial_constructor(__nv_bfloat16))
  V(__has_trivial_constructor(__nv_bfloat162))
}
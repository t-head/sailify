#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");

struct S{
  int v;
  union u2 {
    int n;
    float f;
  };
};
union U {
  int n;
  float f;
  struct s2{
    short a;
    short b;
  };
};
class C{};
class C2;
struct S2 : public C {};

__global__ void kernel() {
  V(!__is_union(S))
  V(__is_union(U))
  V(!__is_union(U&))
  V(__is_union(const U))
  V(!__is_union(const volatile U*))
  V(__is_union(S::u2))
  V(!__is_union(U::s2))
  V(!__is_union(C))
  V(!__is_union(__half))
  V(!__is_union(__half2))
  V(__is_class(C))
  V(__is_class(S))
  V(!__is_class(U))
  V(!__is_class(S::u2))
  V(__is_class(U::s2))
  struct es {} es2;
  V(__is_class(decltype(es2)))
  V(__is_class(C2))
  V(__is_class(__half))
  V(__is_class(__half2))
  V(__is_class(__nv_bfloat16))
  V(__is_class(__nv_bfloat162))
  V(__is_class(dim3))
  V(__is_class(decltype(blockIdx)))
  V(__is_class(decltype(gridDim)))
}

void f() {
  V(!__is_union(S))
  V(__is_union(U))
  V(!__is_union(U&))
  V(__is_union(const U))
  V(!__is_union(const volatile U*))
  V(__is_union(S::u2))
  V(!__is_union(U::s2))
  V(!__is_union(C))
  V(!__is_union(__half))
  V(!__is_union(__half2))
  V(__is_class(C))
  V(__is_class(S))
  V(!__is_class(U))
  V(!__is_class(S::u2))
  V(__is_class(U::s2))
  V(__is_class(S2))
  V(__is_class(C2))
  V(__is_class(__nv_bfloat16))
  V(__is_class(__nv_bfloat162))
  V(__is_class(dim3))
  V(__is_class(decltype(blockIdx)))
  V(__is_class(decltype(gridDim)))
}
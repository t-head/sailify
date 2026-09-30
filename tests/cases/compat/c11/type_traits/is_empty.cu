#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");


union U {};
struct S {};
struct S2 {
  int v;
};
struct S3 {
  static int v;
};
struct S4 {
  virtual void f()=0;
};
struct S5 : S2 {};
struct G {
  int:0;
};
class C {
protected:
  void f() {}
private:
  void f2() {}
};
class C2 : C, S {
  C2(){}
};
class C3 : virtual public C {};

__global__ void kernel() {
  V(!__is_empty(U))
  V(!__is_empty(int))
  V(__is_empty(S))
  V(!__is_empty(S2))
  V(__is_empty(S3))
  V(!__is_empty(S4))
  V(!__is_empty(S5))
  V(__is_empty(G))
  V(__is_empty(C))
  V(__is_empty(C2))
  V(!__is_empty(C3))
}

void f() {
  V(!__is_empty(U))
  V(!__is_empty(int))
  V(__is_empty(S))
  V(!__is_empty(S2))
  V(__is_empty(S3))
  V(!__is_empty(S4))
  V(!__is_empty(S5))
  V(__is_empty(G))
  V(__is_empty(C))
  V(__is_empty(C2))
  V(!__is_empty(C3))
}
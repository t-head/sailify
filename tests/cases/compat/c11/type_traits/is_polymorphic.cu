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

__global__ void kernel() {
  V(!__is_polymorphic(U))
  V(!__is_polymorphic(C))
  V(__is_polymorphic(C2))
  V(__is_polymorphic(C3))
  V(__is_polymorphic(C4))
  V(__is_polymorphic(C5))
  V(!__is_polymorphic(C6))
  V(!__is_polymorphic(__half))
  V(!__is_polymorphic(__half2))
  V(!__is_polymorphic(__nv_bfloat16))
  V(!__is_polymorphic(__nv_bfloat162))
}

void f() {
  V(!__is_polymorphic(U))
  V(!__is_polymorphic(C))
  V(__is_polymorphic(C2))
  V(__is_polymorphic(C3))
  V(__is_polymorphic(C4))
  V(__is_polymorphic(C5))
  V(!__is_polymorphic(C6))
  V(!__is_polymorphic(__half))
  V(!__is_polymorphic(__half2))
  V(!__is_polymorphic(__nv_bfloat16))
  V(!__is_polymorphic(__nv_bfloat162))
}
#define V(x) static_assert(x, "");
#include <cuda_fp16.h>
#include <cuda_bf16.h>
struct A {
  int v;
};
struct B {
  int v;
private:
  int v2;
};
struct B2 : virtual public B {
};
struct B3 : protected B {
};
class C {
public:
  virtual void f() {}
  int v;
};
class D {
  D(const D&) {} 
};
class D2 {
  __device__ D2(const D2&)=default;
};
class D3 {
  D3(const D3&)=delete;
};
class D4 {
  D4()=delete;
};
class D5 {
  ~D5(){}
};
class D6 {
  D6() {}
};
class D7 {
  D7& operator=(const D7& d) {
    return *this;
  }
};
class D8 {
  D8& operator=(D8&& d) {
    return *this;
  }
};
class D9 {
  D9(D9&& d) {}
};

__global__ void kernel() {
  V(__is_trivially_copyable(A))
  V(__is_trivially_copyable(B))
  V(!__is_trivially_copyable(B2))
  V(!__is_trivially_copyable(B2&))
  V(__is_trivially_copyable(B3))
  V(!__is_trivially_copyable(C))
  V(!__is_trivially_copyable(D))
  V(__is_trivially_copyable(D2))
  V(__is_trivially_copyable(D3))
  V(__is_trivially_copyable(D4))
  V(!__is_trivially_copyable(D5))
  V(__is_trivially_copyable(D6))
  V(!__is_trivially_copyable(D7))
  V(!__is_trivially_copyable(D8))
  V(!__is_trivially_copyable(D9))
  V(__is_trivially_copyable(__half))
  V(!__is_trivially_copyable(__half2))
  V(__is_trivially_copyable(__nv_bfloat16))
  V(!__is_trivially_copyable(__nv_bfloat162))
}

void f() {
  V(__is_trivially_copyable(A))
  V(__is_trivially_copyable(B))
  V(!__is_trivially_copyable(B2))
  V(__is_trivially_copyable(B3))
  V(!__is_trivially_copyable(C))
  V(!__is_trivially_copyable(D))
  V(__is_trivially_copyable(D2))
  V(__is_trivially_copyable(D3))
  V(__is_trivially_copyable(D4))
  V(!__is_trivially_copyable(D5))
  V(__is_trivially_copyable(D6))
  V(!__is_trivially_copyable(D7))
  V(!__is_trivially_copyable(D8))
  V(!__is_trivially_copyable(D9))
  V(__is_trivially_copyable(__half))
  V(!__is_trivially_copyable(__half2))
  V(__is_trivially_copyable(__nv_bfloat16))
  V(!__is_trivially_copyable(__nv_bfloat162))
}
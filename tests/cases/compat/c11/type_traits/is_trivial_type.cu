#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");

struct A {
  int v;
  static int n;
  constexpr static float f=1.0f;
};
struct B {
private:
  int v;
};
class C {
  C()=delete;
};
class C2 {
  C2()=default;
  C2(const C2& c)=default;
  C2(C2&& c)=default;
  C2& operator=(const C2&)=default;
  C2& operator=(C2&&)=default;
};
class C3 {
  virtual ~C3();
};
class C4 {
  virtual void f()=0;
};
class C5 {
  __device__ C5() {}
};
class C6 {
  C6(const C6&) {}
};
class C7 {
  C7(C7&&){}
};
class C8 {
  C8& operator=(const C8&) {}
};
class C9 {
  C9& operator=(C9&&) {}
};
class C10 {
  ~C10()=delete;
};
class C11 : virtual public A {};
class D {
public:
  constexpr D(){}
};
class D2 {
public:
  D d;
  int v;
};
class D3 {
private:
  int v;
public:
  D3()=default;
  D3(int x) : v{x} {}
};
class E {
  E()=delete;
  E(const E&)=delete;
  E(E&&)=delete;
  E& operator=(const E&)=delete;
  E& operator=(E&&)=delete;
  ~E()=delete;
};

__global__ void kernel() {
  V(__is_trivial(A))
  V(__is_trivial(B))
  V(__is_trivial(C))
  V(__is_trivial(C2))
  V(!__is_trivial(C3))
  V(!__is_trivial(C4))
  V(!__is_trivial(const C4&))
  V(!__is_trivial(C5))
  V(!__is_trivial(C6))
  V(!__is_trivial(C7))
  V(!__is_trivial(C8))
  V(!__is_trivial(C9))
  V(__is_trivial(C10))
  V(!__is_trivial(C11))
  V(!__is_trivial(D))
  V(!__is_trivial(D2))
  V(__is_trivial(D3))
  V(__is_trivial(__half))
  V(!__is_trivial(__half2))
  V(__is_trivial(__nv_bfloat16))
  V(!__is_trivial(__nv_bfloat162))
  V(__is_trivial(E))
}

void f() {
  V(__is_trivial(A))
  V(__is_trivial(B))
  V(__is_trivial(C))
  V(__is_trivial(C2))
  V(!__is_trivial(C3))
  V(!__is_trivial(C4))
  V(!__is_trivial(const C4&))
  V(!__is_trivial(C5))
  V(!__is_trivial(C6))
  V(!__is_trivial(C7))
  V(!__is_trivial(C8))
  V(!__is_trivial(C9))
  V(__is_trivial(C10))
  V(!__is_trivial(C11))
  V(!__is_trivial(D))
  V(!__is_trivial(D2))
  V(__is_trivial(D3))
  V(__is_trivial(__half))
  V(!__is_trivial(__half2))
  V(__is_trivial(__nv_bfloat16))
  V(!__is_trivial(__nv_bfloat162))
  V(__is_trivial(E))
}
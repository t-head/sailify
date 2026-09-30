#define V(x) static_assert(x, "");
#include <cuda_fp16.h>
#include <cuda_bf16.h>
class A {};
class B : public A {};
class B2 : virtual public A{};
class C {};
class D {
  D()=default;
};
class D2 {
  D2() {}
  D2(const D2&) {}
};
class D3 {
  D3& operator=(const D3&)=delete;
};
class D4 {
  D4(const D4&)=delete;
};
class D5 {
public:
  ~D5() {}
  virtual void f(){}
};
enum day_char {sun, mon, tue, wed, thu, fri, sat};
enum color : char {red=1, blue};
enum class boo {a, b};

__global__ void kernel() {
  V(__is_trivially_assignable(A, B))
  V(__is_trivially_assignable(A*&, B*))
  V(!__is_trivially_assignable(B, A))
  V(__is_trivially_assignable(A, B2))
  V(!__is_trivially_assignable(A, C))
  V(__is_trivially_assignable(A, A))
  V(__is_trivially_assignable(D, D))
  V(__is_trivially_assignable(D2, D2))
  V(!__is_trivially_assignable(D3, D3))
  V(__is_trivially_assignable(D4, D4))
  V(!__is_trivially_assignable(D5, D5))
  V(__is_trivially_assignable(int&, day_char))
  V(!__is_trivially_assignable(day_char&, int))
  V(__is_trivially_assignable(int&, color))
  V(!__is_trivially_assignable(color&, char))
  V(__is_trivially_assignable(boo&, boo))
  V(__is_trivially_assignable(__half, __half))
  V(!__is_trivially_assignable(__half, __half2))
  V(!__is_trivially_assignable(__half2, __half))
  V(!__is_trivially_assignable(__half2, __half2))
  V(__is_trivially_assignable(__nv_bfloat16, __nv_bfloat16))
  V(!__is_trivially_assignable(__nv_bfloat16, __nv_bfloat162))
  V(!__is_trivially_assignable(__nv_bfloat162, __nv_bfloat16))
  V(!__is_trivially_assignable(__nv_bfloat162, __nv_bfloat162))
}

void f() {
  V(__is_trivially_assignable(A, B))
  V(__is_trivially_assignable(A*&, B*))
  V(!__is_trivially_assignable(B, A))
  V(__is_trivially_assignable(A, B2))
  V(!__is_trivially_assignable(A, C))
  V(__is_trivially_assignable(A, A))
  V(__is_trivially_assignable(D, D))
  V(__is_trivially_assignable(D2, D2))
  V(!__is_trivially_assignable(D3, D3))
  V(__is_trivially_assignable(D4, D4))
  V(!__is_trivially_assignable(D5, D5))
  V(__is_trivially_assignable(int&, day_char))
  V(!__is_trivially_assignable(day_char&, int))
  V(__is_trivially_assignable(int&, color))
  V(!__is_trivially_assignable(color&, char))
  V(__is_trivially_assignable(boo&, boo))
  V(__is_trivially_assignable(__half, __half))
  V(!__is_trivially_assignable(__half, __half2))
  V(!__is_trivially_assignable(__half2, __half))
  V(!__is_trivially_assignable(__half2, __half2))
  V(__is_trivially_assignable(__nv_bfloat16, __nv_bfloat16))
  V(!__is_trivially_assignable(__nv_bfloat16, __nv_bfloat162))
  V(!__is_trivially_assignable(__nv_bfloat162, __nv_bfloat16))
  V(!__is_trivially_assignable(__nv_bfloat162, __nv_bfloat162))
}
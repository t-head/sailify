#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");


__device__ int arr[] = {1, 2, 3};
class A {
public:
  void f(int) {}
};
class A2 {
  A2(int x){};
};
class A3 {
  A3()=default;
};
class A4 {
  A4()=delete;
};
class A5 : public A {};
class A6 : virtual public A {};
class A7 : protected A {};
class A8 : private A {};
class A9 {
  ~A9() {}
  A9& operator=(const A9&) {
    return *this;
  }
  A9& operator=(A9&&) {
    return *this;
  }
};
class A10 {
public:
  void (A::*fptr)(int);
};
class B {
public:
  B() {}
};
class B2 : public B {};
class B3 {
  explicit __device__ B3 (int x)  {} 
};
class B4 {
  int i = 0;
  B4() = default;
  explicit B4(B4 const&) = default;
};
class C {
protected:
  int v;
};
class C2 {
private:
  class A* a;
};
class C3 {
public:
  virtual void f() {}
};
class C4 {
protected:
  int v{0};
};
class C5 {
public:
  int v{0};
  __half hf;
private:
  void f() {}
  const static int x = 1;
  static int y;
  constexpr static int z = 2;
};

struct D {
  unsigned a : 16;
  unsigned b : 16;
};
struct D2 {
  unsigned : 8;
  unsigned a : 16;
  unsigned b : 4;
  unsigned : 4;
};
struct E1
{
  int i1;
  E1() : i1(0) { }
};
struct E2 : E1 
{
  using E1::E1;
};
union U {
  int v;
  float f;
};

__global__ void kernel(int *dp) {
  V(__is_aggregate(decltype(arr)))
  V(__is_aggregate(A))
  V(!__is_aggregate(A2))
  V(__is_aggregate(A3))
  V(__is_aggregate(A4))
  V(__is_aggregate(A5))
  V(!__is_aggregate(A6))
  V(!__is_aggregate(A7))
  V(!__is_aggregate(A8))
  V(__is_aggregate(A9))
  V(__is_aggregate(A10))
  V(__is_aggregate(B2))
  V(!__is_aggregate(B3))
  V(!__is_aggregate(B4))
  V(!__is_aggregate(C))
  V(!__is_aggregate(C2))
  V(!__is_aggregate(C3))
  V(!__is_aggregate(C4))
  V(__is_aggregate(C5))
  V(__is_aggregate(D))
  V(__is_aggregate(D2))
  D2 d2 = {0xaabb, 2};
  memcpy(dp, &d2, sizeof(d2));
  V(!__is_aggregate(__half))
  V(!__is_aggregate(__half2))
  V(__is_aggregate(__half_raw))
  V(__is_aggregate(__half2_raw))
  V(!__is_aggregate(__nv_bfloat16))
  V(!__is_aggregate(__nv_bfloat162))
  V(__is_aggregate(__nv_bfloat16_raw))
  V(__is_aggregate(__nv_bfloat162_raw))
  V(__is_aggregate(U))
  __half farr[10];
  V(__is_aggregate(decltype(farr)))
  V(!__is_aggregate(E2))
}

void f() {
  V(__is_aggregate(decltype(arr)))
  V(__is_aggregate(A))
  V(!__is_aggregate(A2))
  V(__is_aggregate(A3))
  V(__is_aggregate(A4))
  V(__is_aggregate(A5))
  V(!__is_aggregate(A6))
  V(!__is_aggregate(A7))
  V(!__is_aggregate(A8))
  V(__is_aggregate(A9))
  V(__is_aggregate(A10))
  V(__is_aggregate(B2))
  V(!__is_aggregate(B3))
  V(!__is_aggregate(B4))
  V(!__is_aggregate(C))
  V(!__is_aggregate(C2))
  V(!__is_aggregate(C3))
  V(!__is_aggregate(C4))
  V(__is_aggregate(C5))
  V(__is_aggregate(D))
  V(!__is_aggregate(__half))
  V(!__is_aggregate(__half2))
  V(__is_aggregate(__half_raw))
  V(__is_aggregate(__half2_raw))
  V(!__is_aggregate(__nv_bfloat16))
  V(!__is_aggregate(__nv_bfloat162))
  V(__is_aggregate(__nv_bfloat16_raw))
  V(__is_aggregate(__nv_bfloat162_raw))
  V(__is_aggregate(U))
  __half farr[10];
  V(__is_aggregate(decltype(farr)))
  V(!__is_aggregate(E2))
}

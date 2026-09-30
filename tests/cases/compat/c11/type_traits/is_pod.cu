#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");

__device__ int arr[] = {1, 2, 3};
class A {};
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
};
class A10 {
  int v;
  A10()=default;
  A10(int x) : v{x} {}
};
class A11 {
  A11();
};
A11::A11()=default;
class A12 {
  A12(const A12 &) {}
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
  int v2;
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
private:
  int v{0};
};
class C5 {
public:
  int a;
protected:
  int b;
};
class C6 {
  int v;
  static C5 c;
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
struct F {};
struct F2 : F {
  int v2;
  F fp;
};
struct F3 : F {
  F fp;
  int v2;
};
struct F4 {
  int v;
  static F3 f;
};
struct F5 : F2 {
  int x;
};
struct F6 : F2 {
  static int x;
};
struct F7: F2 {
  char b : 4;
};
class G {
  G()=delete;
  G(const G&)=delete;
  G(G&&)=delete;
  G& operator=(const G&)=delete;
  G& operator=(G&&)=delete;
  ~G()=delete;
};

__global__ void kernel(int *dp) {
  V(__is_pod(decltype(arr)))
  V(__is_pod(A))
  V(!__is_pod(A2))
  V(__is_pod(A3))
  V(__is_pod(A4))
  V(__is_pod(A5))
  V(!__is_pod(A6))
  V(__is_pod(A7))
  V(__is_pod(A8))
  V(!__is_pod(A9))
  V(__is_pod(A10))
  V(!__is_pod(A11))
  V(!__is_pod(A12))
  V(!__is_pod(B2))
  V(!__is_pod(B3))
  V(!__is_pod(B4))
  V(__is_pod(C))
  V(__is_pod(C2))
  V(!__is_pod(C3))
  V(!__is_pod(C4))
  V(!__is_pod(C5))
  V(__is_pod(C6))
  V(__is_pod(D))
  V(__is_pod(D2))
  D2 d2 = {0xaabb, 2};
  memcpy(dp, &d2, sizeof(d2));
  V(__is_pod(U))
  __half farr[10];
  V(__is_pod(decltype(farr)))
  V(!__is_pod(E2))
  V(__is_pod(F))
  V(__is_pod(F2))
  V(!__is_pod(F3))
  V(__is_pod(F4))
  V(!__is_pod(F5))
  V(__is_pod(F6))
  V(!__is_pod(F7))
  V(__is_pod(__half))
  V(!__is_pod(__half2))
  V(__is_pod(__nv_bfloat16))
  V(!__is_pod(__nv_bfloat162))
  V(__is_pod(G))
}

void f(int *dp) {
  V(__is_pod(decltype(arr)))
  V(__is_pod(A))
  V(!__is_pod(A2))
  V(__is_pod(A3))
  V(__is_pod(A4))
  V(__is_pod(A5))
  V(!__is_pod(A6))
  V(__is_pod(A7))
  V(__is_pod(A8))
  V(!__is_pod(A9))
  V(__is_pod(A10))
  V(!__is_pod(A11))
  V(!__is_pod(A12))
  V(!__is_pod(B2))
  V(!__is_pod(B3))
  V(!__is_pod(B4))
  V(__is_pod(C))
  V(__is_pod(C2))
  V(!__is_pod(C3))
  V(!__is_pod(C4))
  V(!__is_pod(C5))
  V(__is_pod(C6))
  V(__is_pod(D))
  V(__is_pod(D2))
  D2 d2 = {0xaabb, 2};
  memcpy(dp, &d2, sizeof(d2));
  V(__is_pod(U))
  __half farr[10];
  V(__is_pod(decltype(farr)))
  V(!__is_pod(E2))
  V(__is_pod(F))
  V(__is_pod(F2))
  V(!__is_pod(F3))
  V(__is_pod(F4))
  V(!__is_pod(F5))
  V(__is_pod(F6))
  V(!__is_pod(F7))
  V(__is_pod(G))
  V(__is_pod(__half))
  V(!__is_pod(__half2))
  V(__is_pod(__nv_bfloat16))
  V(!__is_pod(__nv_bfloat162))
}

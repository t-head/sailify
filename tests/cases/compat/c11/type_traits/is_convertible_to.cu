#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");
class A {};
class B : public A {};
class B2 : virtual public A {};
class B3 : virtual public A {};
class B4 : public A {};
class C {
public:
    operator A() {
      return A{};
    }
};
class C2 {
protected:
   operator A() {
      return A{};
    }
};
class E {
public:
  template<class T> E(T&&) {}
};
class D {
public:
  D(const A&) {}
};
class D2 {
public:
  D2 operator=(const A&) {
    return D2{};
  }
};
class D3 {
public:
  D3(A&&) {}
};
class D4 {
public:
  D4 operator=(A&&) {
    return D4{};
  }
};
class D5 {
public:
  D5()=delete;
};
class D6 {
public:
  D6(D6&&)=delete;
};
class D7 {
public:
  D7(const D7&)=delete;
};
class D8 {
  D8(const D8&)=default;
};

__global__ void kernel() {
  V(__is_convertible_to(int, double))
  V(__is_convertible_to(int&, int))
  V(!__is_convertible_to(int, int&))
  V(__is_convertible_to(double, int))
  V(__is_convertible_to(A, A))
  V(__is_convertible_to(B, A))
  V(__is_convertible_to(B2, A))
  V(__is_convertible_to(B3, A))
  V(__is_convertible_to(B4, A))
  V(!__is_convertible_to(A, B))
  V(__is_convertible_to(B*, A*))
  V(__is_convertible_to(B4*, A*))
  V(!__is_convertible_to(A*, B*))
  V(__is_convertible_to(B, const A&))
  V(__is_convertible_to(C, A))
  V(__is_convertible_to(C&, A))
  V(!__is_convertible_to(C2, A))
  V(__is_convertible_to(A, E))
  V(__is_convertible_to(const A&, E))
  V(__is_convertible_to(A&&, E))
  V(__is_convertible_to(A*, E))
  V(__is_convertible_to(A, D))
  V(!__is_convertible_to(A, D2))
  V(__is_convertible_to(A, D3))
  V(!__is_convertible_to(A, D4))
  V(__is_convertible_to(__half, __half))
  V(!__is_convertible_to(__half, __half2))
  V(!__is_convertible_to(__half2, __half))
  V(__is_convertible_to(__half2, __half2))
  V(__is_convertible_to(__nv_bfloat16, __nv_bfloat16))
  V(!__is_convertible_to(__nv_bfloat16, __nv_bfloat162))
  V(!__is_convertible_to(__nv_bfloat162, __nv_bfloat16))
  V(__is_convertible_to(__nv_bfloat162, __nv_bfloat162))
  V(__is_convertible_to(D5, D5))
  V(!__is_convertible_to(D6, D6))
  V(!__is_convertible_to(D7, D7))
  V(!__is_convertible_to(D8, D8))
}

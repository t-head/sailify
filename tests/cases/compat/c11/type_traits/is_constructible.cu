#include <cuda_fp16.h>
#include <cuda_bf16.h>
class A;
class B;
class C {
    public:
    C()=default;
    C(int) noexcept {}
    C operator= (const A&)  noexcept ;
    operator B volatile()  noexcept;
};

class A {
public:
  A(){};
  A(int v) {};
  __device__ A (double v) noexcept {} ;
  __host__ A(float v) {};
  A(__half v) noexcept {} ;
  A(__nv_bfloat16 v) noexcept {} ;
  A(short) {};
  __device__ A(__half v, __nv_bfloat162 x)  {};
  __device__ A operator=(const C&) volatile noexcept {
    return A{};
  }
  ~A()=default;
};
class B : public A {
};
C::operator B volatile() noexcept {
    return B{};
}
C C::operator=(const A&)   noexcept {
  return C{};
}

#define V(x) static_assert(x, "");
__global__ void kernel() {
    V(__is_constructible(A, double))
    V(__is_constructible(A, float))
    V(__is_constructible(A, half))
    V(__is_constructible(A))
    V(__is_constructible(A, short))
    V(__is_constructible(A, __half, nv_bfloat162))
    V(__is_destructible(A))
    V(std::is_destructible<A>::value)
    V(std::is_trivially_destructible<A>::value)
    V(__is_nothrow_constructible(A, double))
    V(__is_nothrow_constructible(A, __half))
    V(__is_nothrow_constructible(A, __nv_bfloat16))
    V(__is_nothrow_constructible(C, int))
    V(__is_nothrow_assignable(int&, double))
    V(__is_nothrow_assignable(volatile A, C))
}


__device__ void g() {
    class B {
    public:
    B() noexcept {};
    B(const B&) {}
    B(B&& B) {}
    };
    V(__is_nothrow_constructible(B))
}
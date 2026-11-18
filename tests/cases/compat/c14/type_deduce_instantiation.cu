#include <type_traits>
template <class T>
__device__ auto f(T t) {
    return t; 
}

struct S {
  int v;
  S(int n) : v{n} {}
};

__global__ void kernel() {
    typedef decltype(f(S{2})) S2;
    static_assert(std::is_same<S2, S>::value, "aa");
}

__device__ auto constexpr fn1(int x) {
  return x;
}

template<typename T>
void g() {}

void h() {
  g<decltype(fn1(1))>();
}
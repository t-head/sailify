#include<cuda_fp16.h>
#include<cuda_bf16.h>
#define V(x) static_assert(x, "");

enum COLOR {RED=0, BLUE, GREEN};
enum class BITS {ZERO, ONE};
constexpr int operator "" _int(long double f) {
  return int(f);
}
constexpr COLOR operator ""_c(unsigned long long v) {
  return COLOR(v);
}

template<char c>
constexpr int bin_helper() {
  static_assert(c <= '1', "not a binary digit");
  return c - '0';
}

template<char c, char d, char ...tail>
constexpr int bin_helper() {
  static_assert(c <= '1', "not a binary digit");
  return (c - '0') * (1 << (1 + sizeof...(tail))) + bin_helper<d, tail...>();
}

template<char ...chars>
constexpr int operator "" _b() {
  return bin_helper<chars...>();
}

__global__ void kernel() {
  V(__is_enum(COLOR))
  V(__is_enum(BITS))
  V(__is_enum(decltype(BITS::ZERO)))
  V(__is_enum(decltype(RED)))
  V(!__is_enum(decltype(0)))
  V(!__is_enum(decltype(2.0_int)))
  V(!__is_enum(decltype(101_b)))
  V(__is_enum(decltype(2_c)))
}

void f() {
  V(__is_enum(COLOR))
  V(__is_enum(BITS))
  V(__is_enum(decltype(BITS::ZERO)))
  V(__is_enum(decltype(RED)))
  V(!__is_enum(decltype(0)))
  V(!__is_enum(decltype(2.0_int)))
  V(!__is_enum(decltype(101_b)))
  V(__is_enum(decltype(2_c)))
}
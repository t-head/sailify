#include "test_buf.h"

template<typename T, size_t n>
struct AtomicAddImpl;

template<typename T>
struct AtomicAddImpl<T, 4> {
  inline __device__ void operator() (T* addr, T val) {
    atomicAdd(addr, val);
  }
};
template<typename T>
struct AtomicAddImpl<T, 8> {
  inline __device__ void operator() (T* addr, T val) {
    unsigned long long* addr_as_ui = (unsigned long long *)addr;
    unsigned long long old = *addr_as_ui;
    unsigned long long newval, assumed;
    do {
      assumed = old;
      newval = val + (T)old;
      old = atomicCAS(addr_as_ui, assumed, newval);
    } while (assumed != old);
  }
};
template<typename T>
struct AtomicAddImpl<T, 1> {
  inline __device__ void operator() (T* addr, T val) {
    size_t offset = (size_t)addr & 3;
    uint32_t *addr_as_ui = (uint32_t*)((char*)addr - offset);
    uint32_t old = *addr_as_ui;
    uint32_t shift = offset * 8;
    uint32_t old_byte, newval, assumed;
    do {
      assumed = old;
      old_byte = (old >> shift) & 0xff;
      newval = old_byte + val;
      newval = (old & ~(0xff << shift)) | (newval << shift);
      old = atomicCAS(addr_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

static inline __device__ void gpuAtomicAdd(int* addr, int val) {
  AtomicAddImpl<int, sizeof(int)>()(addr, val);
}
static inline __device__ void gpuAtomicAdd(unsigned long long* addr, unsigned long long val) {
  AtomicAddImpl<unsigned long long, sizeof(unsigned long long)>()(addr, val);
}
static inline __device__ void gpuAtomicAdd(uint8_t* addr, uint8_t val) {
  AtomicAddImpl<uint8_t, sizeof(uint8_t)>()(addr, val);
}

class ReduceAdd {
public:
  template<typename scalar_t>
  constexpr __device__ void operator() (scalar_t* input, scalar_t* output) const {
    gpuAtomicAdd(output, input[0]);
  }
};
static ReduceAdd reduce_add;

class ReduceSub {
public:
  template<typename scalar_t>
  constexpr __device__ void operator() (scalar_t* input, scalar_t* output) const {
    atomicSub(output, input[0]);
  }
};
static ReduceSub reduce_sub;

template<typename func_t>
__global__ void kernel(func_t f) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  f(idx);
}

template<typename func_t>
static void kernel_helper(const func_t& f) {
  kernel<func_t><<<2, 32>>>(f);
}

enum dtype {kint, kfloat, kbool};

struct scalar_wrapper {
  dtype dt;
  char* p;
  scalar_wrapper(dtype dt, char* p) : dt(dt), p(p) {}
  dtype dtype() {
    return dt;
  }
};
template<typename T>
struct type_to_enum {};
template<>
struct type_to_enum<int> {
  const static dtype value = kint;
};
template<>
struct type_to_enum<float> {
  const static dtype value = kfloat;
};
template<>
struct type_to_enum<uint8_t> {
  const static dtype value = kbool;
};

template<bool is_add, typename scalar_t>
struct internal_launch_kernel_helper {
  template<typename func_t>
  void operator() (
    scalar_wrapper& input, scalar_wrapper& output,
    const func_t& f, int size
  ) {
    if (size > 64) {
      scalar_wrapper sub_input = scalar_wrapper(
        input.dtype(), input.p + sizeof(scalar_t) * 64);
      internal_launch_kernel_helper<is_add, scalar_t>()(
        sub_input, output, f, size - 64
      );
    }
    auto loop = [=]__device__(int i) {
      f((scalar_t*)(input.p) + i, (scalar_t*)(output.p));
    };
    kernel_helper(loop);
  }
};

int test_wrapper1() {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();
  scalar_wrapper input(kint, (char*)dinput), output(kint, (char*)doutput);
  
  internal_launch_kernel_helper<true, int>()(
    input, output, reduce_add, 128
  );
  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, member_template1) {
  EXPECT_TRUE(test_wrapper1());
}

template<dtype dt>
struct type_wrapper {};
template<>
struct type_wrapper<kint> {
  using type = int;
  static type t;
};
template<>
struct type_wrapper<kfloat> {
  using type = float;
  static type t;
};
template<>
struct type_wrapper<kbool> {
  using type = uint8_t;
  static type t;
};
template<int N>
struct alignas(N) opaqueType {
  char data[N];
};

template<bool is_add=true, bool is_opaque=false>
struct base_launch_kernel_helper {
  template<typename func_t>
  void operator() (
    scalar_wrapper& input, scalar_wrapper& output,
    const func_t& f, int size
  ) {
    [&] {
      const auto& the_type = input.dtype();
      switch (the_type) {
        case kint:
            [&] {
              using dtype = typename std::conditional<is_opaque,
                opaqueType<sizeof(int)>, type_wrapper<kint>::type>::type;
              internal_launch_kernel_helper<is_add, dtype>()(
              input, output, f, size);
            }();
          break;
        default:
          assert(0);
          break;
      }
    }();
  }
  void operator() (
    scalar_wrapper& input, scalar_wrapper& output,
    const ReduceSub& f, int size
  ) {
    [&] {
      switch (input.dtype()) {
        case kint:
          { using dtype = type_wrapper<kint>::type;
            [&] { internal_launch_kernel_helper<is_add, dtype>()(
              input, output, f, size
            );}();
          break;}
        default:
          assert(0);
          break;
      }
    }();
  }
};

int test_wrapper2() {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();
  scalar_wrapper input(kint, (char*)dinput), output(kint, (char*)doutput);
  
  base_launch_kernel_helper<true>()(
    input, output, reduce_add, 128
  );
  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, member_template2) {
  EXPECT_TRUE(test_wrapper2());
}

enum reduce_type {REDUCE_ADD, REDUCE_SUB};

void test_reduce_add(scalar_wrapper& input, scalar_wrapper& output, const reduce_type& reduce, int size) {
  switch (reduce) {
    case REDUCE_ADD:
      base_launch_kernel_helper<>()(input, output, reduce_add, size);
      break;
    case REDUCE_SUB:
      base_launch_kernel_helper<false>()(input, output, reduce_sub, size);
      break;
    default:
      assert(0);
      break;
  }
}

int test_wrapper3() {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();
  scalar_wrapper input(kint, (char*)dinput), output(kint, (char*)doutput);
  
  test_reduce_add(input, output, REDUCE_ADD, 128);
  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, member_template3) {
  EXPECT_TRUE(test_wrapper3());
}

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \
    using scalar_t = type;                         \
    return __VA_ARGS__();                          \
  }

#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, ...) \
  [&] {                                                                 \
    switch (TYPE) {                                                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1,                                 \
        decltype(type_wrapper<SCALARTYPE1>::t),                         \
        __VA_ARGS__)                                                    \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2,                                 \
        decltype(type_wrapper<SCALARTYPE2>::t),                         \
        __VA_ARGS__)                                                    \
      default:                                                          \
        assert(0);                                                      \
    }                                                                   \
  }()

  template<bool is_add=true, bool is_opaque=true>
  struct macro_launch_kernel_helper {
    template<typename func_t>
    void operator() (
      scalar_wrapper& input, scalar_wrapper& output,
      const func_t& f, int size
    ) {
      AT_DISPATCH_ALL_TYPES_AND2(
        kint, kbool, input.dtype(), [&] {
          using dtype = typename std::conditional<is_opaque, opaqueType<sizeof(scalar_t)>, scalar_t>::type;
          auto input2 = scalar_wrapper(type_to_enum<dtype>::value, input.p);
          auto output2 = scalar_wrapper(type_to_enum<dtype>::value, output.p);
          internal_launch_kernel_helper<is_add, dtype>()(input2, output2, f, size);
        }
      );
    }
    void operator() (
      scalar_wrapper& input, scalar_wrapper& output,
      const ReduceSub& f, int size
    ) {
      AT_DISPATCH_ALL_TYPES_AND2(
        kint, kbool, input.dtype(), [&] {
          using dtype = typename std::conditional<is_opaque, opaqueType<sizeof(scalar_t)>, scalar_t>::type;
          auto input2 = scalar_wrapper(type_to_enum<dtype>::value, input.p);
          auto output2 = scalar_wrapper(type_to_enum<dtype>::value, output.p);
          internal_launch_kernel_helper<is_add, dtype>()(input2, output2, f, size);
        }
      );
    }
  };

  void test_reduce_add_macro(scalar_wrapper& input, scalar_wrapper& output, int size) {
    macro_launch_kernel_helper<true, false>()(input, output, reduce_add, size);
  }

  int test_wrapper4() {
    auto testDBufInput = testBuf<int, memory_scope::device>(128);
    auto testDBufOutput = testBuf<int, memory_scope::device>(1);
    auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
    testDBufInput.initialize([](int i) {return i;});
    auto dinput = testDBufInput.get_pointer();
    auto doutput = testDBufOutput.get_initialized_pointer();
    scalar_wrapper input(kint, (char*)dinput), output(kint, (char*)doutput);
    
    test_reduce_add_macro(input, output, 128);
    testHBufOuptut = testDBufOutput;
    bool pass = (testHBufOuptut[0] == 8128);
    if (pass) {
      TLOG_INFO << "Test result verify pass !\n";
    } else {
      TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
    }
    return pass;
  }
  
  TEST(extended_lambda, member_template4) {
    EXPECT_TRUE(test_wrapper4());
  }
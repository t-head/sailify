#include "test_buf.h"

template<int size, typename R>
__global__ void kernel(R reduction) {
  reduction.template run<size>();
}

struct ReduceConfig {
  int size = 1;
  ReduceConfig(int size) : size(size) {}
  dim3 block() const {
    return dim3(128, 1);
  }
  dim3 grid() const {
    return dim3(size / 128, 1);
  }
};

template<int size, typename R>
static void launch_kernel(const ReduceConfig& config, const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();
  kernel<size, R><<<grid, block>>>(reduction);
}

template<typename scalar_t, typename ops_t>
struct ReduceOp {
  ops_t ops;
  const scalar_t* src;
  scalar_t* dst;
  ReduceOp(ops_t ops, scalar_t* src, scalar_t* dst) : ops(ops), src(src), dst(dst) {}

  template<int size>
  __device__ void run() const {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
      scalar_t ret = 0;
      for (int i = 0; i < size; i ++) {
        ret = ops(ret, src[i]);
      }
      dst[0] = ret;
    }
  }
};

template<int size, typename scalar_t, typename ops_t>
inline void gpu_reduce_kernel(scalar_t* dinput, scalar_t* doutput, const ops_t& ops) {
  auto config = ReduceConfig(size);
  auto reduce = ReduceOp<scalar_t, ops_t>(ops, dinput, doutput);
  launch_kernel<size>(config, reduce);
}

int test_wrapper() {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();

  gpu_reduce_kernel<128>(dinput, doutput, []__host__ __device__(int a, int b) -> int {
    return a + b;
  });

  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, template_functor) {
  EXPECT_TRUE(test_wrapper());
}

template<typename out_t, typename func_t>
struct func_wrapper_t {
  func_t combine;
  func_wrapper_t(const func_t& op) : combine(op) {}
  __device__ out_t operator()(out_t a, out_t b) const {
    return combine(a, b);
  }
};
template<typename out_t, typename func_t>
func_wrapper_t<out_t, func_t> func_wrapper(const func_t& op) {
  return func_wrapper_t<out_t, func_t> {op};
}

int test_wrapper2() {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();

  gpu_reduce_kernel<128>(dinput, doutput, func_wrapper<int>([]__host__ __device__(int a, int b) -> int {
    return a + b;
  }));

  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, template_functor2) {
  EXPECT_TRUE(test_wrapper2());
}

template<typename scalar_t, int size, typename acc_t = scalar_t>
struct sum_functor {
  using out_t = scalar_t;
  void operator()(scalar_t* dinput, scalar_t* doutput) {
    gpu_reduce_kernel<size>(dinput, doutput, func_wrapper<out_t>([] __host__ __device__(acc_t a, acc_t b) -> out_t {
      return a + b;
    }));
  }
};

int test_wrapper3(int size) {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();

  [&] {
    switch (size) {
      case 128:
        [&]() {
          sum_functor<int, 128>{}(dinput, doutput);
        }();
        break;
      default:
        assert(0);
        break;
    }
  }();

  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, template_functor3) {
  EXPECT_TRUE(test_wrapper3(128));
}

enum dtype {kint, kfloat};

struct scalar_wrapper {
  dtype dt;
  char* p;
  scalar_wrapper(dtype dt) : dt(dt) {}
  dtype dtype() {
    return dt;
  }
};

template<
  template<
    typename scalar_t,
    int size,
    typename acc_t = scalar_t>
  typename OpFunctor,
  typename GeneralDispatcher>
static void test_dispatch(scalar_wrapper& input, scalar_wrapper& output, GeneralDispatcher op) {
  if (input.dtype() == kfloat) {
    return OpFunctor<float, 128>{}((float*)(input.p), (float*)(output.p));
  }
  op(input, output);
}

static void sum_kernel_cuda(scalar_wrapper& input, scalar_wrapper& output) {
  auto general_dispatcher = [](scalar_wrapper& input, scalar_wrapper& output) {
    [&] {
      switch (input.dtype()) {
        case kint:
          [&]() {
            sum_functor<int, 128>{}((int*)(input.p), (int*)(output.p));
          }();
          break;
        default:
          assert(0);
          break;
      }
    }();
  };
  test_dispatch<sum_functor>(input, output, general_dispatcher);
}

int test_wrapper4(int size) {
  auto testDBufInput = testBuf<int, memory_scope::device>(128);
  auto testDBufOutput = testBuf<int, memory_scope::device>(1);
  auto testHBufOuptut = testBuf<int, memory_scope::host>(1);
  testDBufInput.initialize([](int i) {return i;});
  auto dinput = testDBufInput.get_pointer();
  auto doutput = testDBufOutput.get_initialized_pointer();

  scalar_wrapper input(kint), output(kint);
  input.p = (char*)dinput;
  output.p = (char*)doutput;
  sum_kernel_cuda(input, output);
  testHBufOuptut = testDBufOutput;
  bool pass = (testHBufOuptut[0] == 8128);
  if (pass) {
    TLOG_INFO << "Test result verify pass !\n";
  } else {
    TLOG_ERROR << "Test result verify fail ! Excepted value: 8128, actual value: " << testHBufOuptut[0] << "\n";
  }
  return pass;
}

TEST(extended_lambda, template_functor4) {
  EXPECT_TRUE(test_wrapper4(128));
}
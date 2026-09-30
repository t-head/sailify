#include "test_buf.h"

template<typename F1, typename F2>
__global__ void kernel(int* dp, F1 f1, F2 f2) {
  int tid = threadIdx.x;
  dp[tid] = (tid % 2) ? f2(tid) : f1(tid);
}

template<typename F1, typename F2>
static void kernel_helper(int* dp, const F1& f1, const F2& f2) {
  kernel<F1, F2><<<1, 32>>>(dp, f1, f2);
}

template<typename F1, typename F2>
void test_kernel(int* dp, size_t size, const F1& f1, const F2& f2) {
  if (size > 64) {
    for (int l = 0; l < size / 64; l ++) {
      test_kernel(dp + 64 * l, 64, f1, f2);
    }
    return;
  }
  for (int l = 0; l < size / 32; l ++) {
    kernel_helper(dp + 32 * l, [=]__device__(int v) {
      return f1(v + l); 
    }, [=]__device__(int v) {
      return f2(v + l);
    });
  }
}

template<typename T>
void test_kernel_size_impl(int* dp) {
  test_kernel(dp, sizeof(T), [=]__device__(int v) {
    return v + sizeof(T) / 128;
  }, []__device__(int v) {
    return 2 * v;
  });
}

template<size_t SIZE>
struct size_helper {
  char buf[SIZE];
};

void test_sizes(int*dp, size_t size) {
  [&] {
    switch (size) {
      case 64:
        {using type = size_helper<64>;
        [&] { test_kernel_size_impl<type>(dp); }();
        break;}
      case 128:
        {using type = size_helper<128>;
        [&] { test_kernel_size_impl<type>(dp); }();
        break;}
      default:
        assert(0);
        break;
    }
  }();
}

void get_goldens(int* hp, size_t size) {
  for (int l = 0; l < size / 32; l ++) {
    int v = l % 2;
    for (int k = 0; k < 32; k ++) {
      hp[l * 32 + k] = (k % 2) ? 2 * (v + k) : 1 + (v + k);
    }
  }
}

TEST(extended_lambda, scope_tag) {
  auto testDBuf = testBuf<int, memory_scope::device>(128);
  auto testHBuf = testBuf<int, memory_scope::host>(128);
  auto testHGBuf = testBuf<int, memory_scope::host>(128);
  auto dp = testDBuf.get_initialized_pointer();
  test_sizes(dp, 128);
  testHBuf = testDBuf;
  get_goldens(testHGBuf.get_pointer(), 128);
  bool pass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = testHGBuf[i];
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running result: " << (pass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(pass);
}
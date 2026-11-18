#include "test_buf.h"

template<typename T1, typename T2, typename T3, typename T4>
__global__ void kernel(int *dp, T1 lam, T2 lam1, T3 lam2, T4 lam3) {
  dp[0] = lam();
  dp[1] = lam1();
  dp[2] = lam2();
  dp[3] = lam3();
}

struct S {
  int v;
  S(int n) : v{n} {}
  S()=default;
};

void use_lam(int &hp, int *dp) {
  auto lam = [hp]__device__{return hp * 2;};
  auto lam1 = [x = 5] __device__ () { return x; };
  int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto lam2 = [a]__device__ {return a[8] + a[9];};
  S ss[5] = {1, 2, 3, 4, 5};
  auto lam3 = [ss]__device__{return ss[3].v + ss[4].v;};
  kernel<<<1, 1>>>(dp, lam, lam1, lam2, lam3);
}

TEST(extended_lambda, capture_variable) {
  auto testDBuf = testBuf<int, memory_scope::device>(4);
  auto testHBuf = testBuf<int, memory_scope::host>(4);
  testDBuf.clear();
  int h = 10;
  use_lam(h, testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 20);
  EXPECT_EQ(testHBuf[1], 5);
  EXPECT_EQ(testHBuf[2], 17);
  EXPECT_EQ(testHBuf[3], 9);
}

__device__ void f(int *dp) {
  struct S {
      int v;
      __device__ __host__ S(int n) : v{n} {}
      __device__ void run(int m) {
          auto lam = [this] (int n){
              this->v += n;
          };
          lam(m);
      }
  };
  S ss{10};
  ss.run(7);
  dp[0] = ss.v;
}

__global__ void kernel2(int *dp) {
  f(dp);
}

TEST(extended_lambda, capture_this) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel2<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 17);
}

#define HD __host__ __device__

HD void f2(int* dp, int index, int& a, int& b) {
  [&a, b, index](int* p) {p[index] = (++a) + b;}(dp);
}
__global__ void kernel3(int* dp) {
  int a = threadIdx.x;
  int b = blockIdx.x + 1;
  f2(dp, threadIdx.x, a, b);
  dp[threadIdx.x] += a;
}
void host3(int* hp, int size) {
  int a = 0; int b = 1;
  for (int k = 0; k < size; k ++) {
    f2(hp, k, a, b);
  }
  for (int k = 0; k < size; k ++) {
    hp[k] += a;
  }
}

TEST(extended_lambda, capture_modify) {
  auto testDBuf = testBuf<int, memory_scope::device>(8);
  auto testHBuf = testBuf<int, memory_scope::host>(8);
  auto dp = testDBuf.get_initialized_pointer();
  kernel3<<<1, 8>>>(dp);
  testHBuf = testDBuf;
  int dgolden[] = {3, 5, 7, 9, 11, 13, 15, 17};
  bool dpass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = dgolden[i];
    bool check = (golden == v);
    dpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running device result: " << (dpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(dpass);
  auto hp = testHBuf.get_initialized_pointer();
  host3(hp, 8);
  bool hpass = true;
  int hgolden[] = {10, 11, 12, 13, 14, 15, 16, 17};
  testHBuf.foreach([&](int v, int i) {
    int golden = hgolden[i];
    bool check = (golden == v);
    hpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running host result: " << (hpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(hpass);
}

HD void f3(int* dp, int index, int& a, int& b) {
  [&, index](int* p) {p[index] = (++a) + (b++);}(dp);
}
__global__ void kernel4(int* dp) {
  int a = threadIdx.x;
  int b = blockIdx.x + 1;
  f3(dp, threadIdx.x, a, b);
  dp[threadIdx.x] += a + b;
}
void host4(int* hp, int size) {
  int a = 0; int b = 1;
  for (int k = 0; k < size; k ++) {
    f3(hp, k, a, b);
  }
  for (int k = 0; k < size; k ++) {
    hp[k] += a + b;
  }
}

TEST(extended_lambda, capture_ref_default) {
  auto testDBuf = testBuf<int, memory_scope::device>(8);
  auto testHBuf = testBuf<int, memory_scope::host>(8);
  auto dp = testDBuf.get_initialized_pointer();
  kernel4<<<1, 8>>>(dp);
  testHBuf = testDBuf;
  int dgolden[] = {5, 7, 9, 11, 13, 15, 17, 19};
  bool dpass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = dgolden[i];
    bool check = (golden == v);
    dpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running device result: " << (dpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(dpass);
  auto hp = testHBuf.get_initialized_pointer();
  host4(hp, 8);
  bool hpass = true;
  int hgolden[] = {19, 21, 23, 25, 27, 29, 31, 33};
  testHBuf.foreach([&](int v, int i) {
    int golden = hgolden[i];
    bool check = (golden == v);
    hpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running host result: " << (hpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(hpass);
}

HD void f4(int* dp, int index, int& a, int& b) {
  [=, &a, &b](int* p) {p[index] = (++a) + (b++);}(dp);
}
__global__ void kernel5(int* dp) {
  int a = threadIdx.x;
  int b = blockIdx.x + 1;
  f4(dp, threadIdx.x, a, b);
  dp[threadIdx.x] += a + b;
}
void host5(int* hp, int size) {
  int a = 0; int b = 1;
  for (int k = 0; k < size; k ++) {
    f4(hp, k, a, b);
  }
  for (int k = 0; k < size; k ++) {
    hp[k] += a + b;
  }
}

TEST(extended_lambda, capture_value_default) {
  auto testDBuf = testBuf<int, memory_scope::device>(8);
  auto testHBuf = testBuf<int, memory_scope::host>(8);
  auto dp = testDBuf.get_initialized_pointer();
  kernel5<<<1, 8>>>(dp);
  testHBuf = testDBuf;
  int dgolden[] = {5, 7, 9, 11, 13, 15, 17, 19};
  bool dpass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = dgolden[i];
    bool check = (golden == v);
    dpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running device result: " << (dpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(dpass);
  auto hp = testHBuf.get_initialized_pointer();
  host5(hp, 8);
  bool hpass = true;
  int hgolden[] = {19, 21, 23, 25, 27, 29, 31, 33};
  testHBuf.foreach([&](int v, int i) {
    int golden = hgolden[i];
    bool check = (golden == v);
    hpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running host result: " << (hpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(hpass);
}

template<typename T, int SIZE>
struct tbuf {
  T data[SIZE];
};

template<typename T>
HD void f_auto(T *dp) {
  tbuf<T, 16> mem;
  [](auto& mem) {
    mem.data[0] = T(1);
    mem.data[15] = T(15);
  } (mem);
  [&](auto&& p) {
#ifdef __CUDA_ARCH__
    atomicAdd(p, mem.data[0] + mem.data[15]);
#else
    p[0] += mem.data[0] + mem.data[15];
#endif
  } (dp);
}

template<typename T>
__global__ void kernel_auto(T* dp) {
  f_auto(dp);
}
template<typename T>
void host_auto(T* hp) {
  f_auto(hp);
}

TEST(extended_lambda, auto_arg) {
  auto testDBufF = testBuf<float, memory_scope::device>(1);
  auto testHBufF = testBuf<float, memory_scope::host>(1);
  auto dpf = testDBufF.get_initialized_pointer();
  TLOG_INFO << "Testing with float type...\n";
  kernel_auto<<<5, 1>>>(dpf);
  testHBufF = testDBufF;
  bool fpass = (testHBufF[0] == 80.0f);
  if (!fpass) {
    TLOG_ERROR << "Expected float device result: 80, actual value: " << testHBufF[0] << "\n";
  } else {
    TLOG_INFO << "Checking float device result pass !\n";
  }
  EXPECT_TRUE(fpass);
  auto hpf = testHBufF.get_initialized_pointer();
  host_auto(hpf);
  bool hfpass = (testHBufF[0] == 16.0f);
  if (!hfpass) {
    TLOG_ERROR << "Expected float host result: 16, actual value: " << testHBufF[0] << "\n";
  } else {
    TLOG_INFO << "Checking float host result pass !\n";
  }
  EXPECT_TRUE(hfpass);

  TLOG_INFO << "Testing with ULL type...\n";
  auto testDBufL = testBuf<unsigned long long, memory_scope::device>(1);
  auto testHBufL = testBuf<unsigned long long, memory_scope::host>(1);
  auto dpl = testDBufL.get_initialized_pointer();
  kernel_auto<<<5, 1>>>(dpl);
  testHBufL = testDBufL;
  bool upass = (testHBufL[0] == 80);
  if (!upass) {
    TLOG_ERROR << "Expected ull device result: 80, actual value: " << testHBufL[0] << "\n";
  } else {
    TLOG_INFO << "Checking ull device result pass !\n";
  }
  EXPECT_TRUE(upass);
  auto hpl = testHBufL.get_initialized_pointer();
  host_auto(hpl);
  bool hupass = (testHBufL[0] == 16);
  if (!hupass) {
    TLOG_ERROR << "Expected ull host result: 16, actual value: " << testHBufL[0] << "\n";
  } else {
    TLOG_INFO << "Checking ull host result pass !\n";
  }
  EXPECT_TRUE(hupass);
}

__device__ int capture_ref_init(int &v) {
  int a = 5;
  [&v=a]() {
    v += 100;
  }();
  return a;
}

__global__ void kernel_capture_ref_init(int *dp) {
  auto v = capture_ref_init(dp[0]);
  dp[1] = v;
}

TEST(extended_lambda, capture_ref_init) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_capture_ref_init<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 0);
  EXPECT_EQ(testHBuf[1], 105);
}

struct lambda_wrapper
{
    __host__ __device__ lambda_wrapper(int v1, int v2) : x(v1), y(v2) {}
    int x, y;
    __device__ __host__ int operator()(int v) { return x += v; }
    __device__ __host__ int f()
    {
        return [=]() -> int
        {
            return operator()(this->x + y);
        }();
    }
};

__global__ void kernel_lambda_wrapper(int *dp) {
  auto lam = lambda_wrapper(1, 2);
  dp[0] = [=] __host__ __device__ () mutable { return lam.f();}();
}

TEST(extended_lambda, capture_lambda_wrapper) {
  auto testDBuf = testBuf<int, memory_scope::device>(1);
  auto testHBuf = testBuf<int, memory_scope::host>(1);
  kernel_lambda_wrapper<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 4);
}

struct lambda_wrapper2
{
    __host__ __device__ lambda_wrapper2(int v1, int v2) : x(v1), y(v2) {}
    int x, y;
    __device__ __host__ int operator()(int v) { return x += v; }
    __device__ __host__ int f()
    {
        return [=]() -> int
        {
            return operator()((*this).x += y);
        }();
    }
};

__global__ void kernel_lambda_wrapper2(int *dp) {
  auto lam = lambda_wrapper2(1, 2);
  dp[0] = [&] __host__ __device__ () mutable { return lam.f();}();
  dp[1] = lam.x;
  auto lam2 = lambda_wrapper2(1, 2);
  dp[2] = [=] __host__ __device__ () mutable { return lam2.f();}();
  dp[3] = lam2.x;
}

TEST(extended_lambda, capture_lambda_wrapper2) {
  auto testDBuf = testBuf<int, memory_scope::device>(4);
  auto testHBuf = testBuf<int, memory_scope::host>(4);
  kernel_lambda_wrapper2<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 6);
  EXPECT_EQ(testHBuf[1], 6);
  EXPECT_EQ(testHBuf[2], 6);
  EXPECT_EQ(testHBuf[3], 1);
}

template<typename T>
__device__ T lam_pack_helper(T arg) {
  return arg;
}
template<typename T, typename ...Args>
__device__ auto lam_pack_helper(T arg, Args... args) {
  return lam_pack_helper(arg) + lam_pack_helper(args...);
}

template<typename ...Args>
__device__ auto lam_pack(Args... args) {
  return [args...] { return lam_pack_helper(args...);}();
}

template<typename ...Args>
__device__ void lam_pack2(Args&... args) {
  [&args...] {
    ((++args), ...);
  }();
}

__global__ void kernel_lambda_pack(int* dp) {
  dp[0] = lam_pack(1, 2, 3, 4);
  dp[1] = lam_pack(1.0f, 2.0, 3, 4);
  int a = 1, b = 2, c = 3;
  lam_pack2(a, b, c);
  dp[2] = a + b + c;
}

TEST(extended_lambda, capture_lambda_pack) {
  auto testDBuf = testBuf<int, memory_scope::device>(3);
  auto testHBuf = testBuf<int, memory_scope::host>(3);
  kernel_lambda_pack<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 10);
  EXPECT_EQ(testHBuf[1], 10);
  EXPECT_EQ(testHBuf[2], 9);
}

struct capture_star_this {
  int v;
  __device__ capture_star_this(int v) : v{v} {}
  __device__ int copy() {
    return [copy=*this] __device__ __host__ () mutable {return ++copy.v;}();
  }
};

__global__ void kernel_capture_star_this(int* dp) {
  auto ss = capture_star_this(100);
  dp[0] = ss.copy();
  dp[1] = ss.v;
}

TEST(extended_lambda, capture_lambda_star_this) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_capture_star_this<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 101);
  EXPECT_EQ(testHBuf[1], 100);
}

__device__ int v = 10;
__device__ int vv = 20;
struct capture_member_initializer {
  int v;
  __device__ capture_member_initializer(int v) : v{v} {}
  int v2{[=]__device__ () -> int {return 2 * v;}()};
  int v3{[=]__device__ () -> int  {return 1 + vv;}()};
};

__global__ void kernel_cmi(int *dp) {
  capture_member_initializer cmi(100);
  dp[0] = cmi.v2;
  dp[1] = cmi.v3;
}

TEST(extended_lambda, capture_lambda_member_initializer) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_cmi<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 200);
  EXPECT_EQ(testHBuf[1], 21);
}

__global__ void kernel_name_lookup(int *dp) {
  int x = 4;
  auto y = [&r = x, x = x + 1]() -> int
  {
      r += 2;
      return x * x;
  }();
  dp[0] = y;
  dp[1] = x;
}

TEST(extended_lambda, capture_lambda_name_lookup) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_name_lookup<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 25);
  EXPECT_EQ(testHBuf[1], 6);
}

__global__ void kernel_capture_direct_ref(int* dp) {
  __shared__ int tsm;
  if (threadIdx.x == 0) {
    tsm = 100;
  }
  __syncthreads();
  int &ref = tsm;
  auto lam = [&] { return dp[threadIdx.x * 2] = ref; };
  __syncthreads();
  if (threadIdx.x == 0) {
    tsm = 200;
  }
  __syncthreads();
  dp[threadIdx.x * 2 + 1] = lam();
}

TEST(extended_lambda, capture_lambda_direct_ref) {
  auto testDBuf = testBuf<int, memory_scope::device>(128);
  auto testHBuf = testBuf<int, memory_scope::host>(128);
  kernel_capture_direct_ref<<<1, 64>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  bool dpass = true;
  testHBuf.foreach([&](int v, int i) {
    int golden = 200;
    bool check = (golden == v);
    dpass &= check;
    if (!check) {
      TLOG_ERROR << "Mismatched index: " << i << ", val = "
        << v << ", golden = " << golden << "\n";
    }
  });
  TLOG_INFO << "Running device result: " << (dpass ? "pass" : "fail") << "\n";
  EXPECT_TRUE(dpass);
}

class capture_member
{
  int x = 0;
public:
  __device__ void f(int* dp)
  {
      int i = 0;
      auto l2 = [i, x = x, dp] { dp[0] = i * 10 + x; };
      i = 1; x = 1; l2();
      auto l3 = [i, &x = x, dp] { dp[1] = i * 10 + x; };
      i = 2; x = 2; l3();
  }
};

__global__ void kernel_capture_member(int* dp) {
  capture_member cpm;
  cpm.f(dp);
}

TEST(extended_lambda, capture_lambda_data_member) {
  auto testDBuf = testBuf<int, memory_scope::device>(2);
  auto testHBuf = testBuf<int, memory_scope::host>(2);
  kernel_capture_member<<<1, 1>>>(testDBuf.get_initialized_pointer());
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 0);
  EXPECT_EQ(testHBuf[1], 12);
}
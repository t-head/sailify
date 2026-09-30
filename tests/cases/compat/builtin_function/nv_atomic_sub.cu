#include "test_buf.h"

#define HD __host__ __device__

template<typename T>
HD T aug_value(T val) {
  return val;
}
template<>
HD uint32_t aug_value(uint32_t val) {
  return (val << 16) | val;
}
template<>
HD int aug_value(int val) {
  return (val << 16) | val;
}
template<>
HD uint64_t aug_value(uint64_t val) {
  return (val << 32) | val;
}

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_sub(T *dst) {
  int tid = threadIdx.x;
  T value = aug_value(T(tid));
  __nv_atomic_sub(dst, value, order, scope);
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_sub() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_sub with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  testDDst = testHDst.initialize([](int i) {
    T v = (127 * 128 / 2) + 1;
    return aug_value(v);
  });

  kernel_nv_atomic_sub<T, order, scope><<<1, 128>>>(testDDst);
  testHDst = testDDst;

  T golden = 1;
  golden = aug_value(golden);
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }

  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_sub_u32) {
  bool ret = true;
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_sub_i32) {
  bool ret = true;
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_sub_u64) {
  bool ret = true;
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_sub_f32) {
  bool ret = true;
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_sub_f64) {
  bool ret = true;
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_fetch_sub(T *dst, T *rtn) {
  int tid = threadIdx.x;
  rtn[tid] = __nv_atomic_fetch_sub(dst, 1, order, scope);
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_fetch_sub() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_fetch_sub with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  auto testHRtn = testBuf<T, memory_scope::host>(128);
  auto testDRtn = testBuf<T, memory_scope::device>(128);
  testDDst = testHDst.initialize([](int i) { return 129; });

  kernel_nv_atomic_fetch_sub<T, order, scope><<<1, 128>>>(testDDst, testDRtn);
  testHDst = testDDst;
  (testHRtn = testDRtn).sort();

  T golden = 1;
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }
  TLOG_INFO << "Verifying rtn values ...\n";
  testHRtn.foreach([&](T v, int i) {
    T golden = i + 2;
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Idx: " << i << ", golden: " << golden << ", value: "
        << v << "\n";
    }
  });

  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fetch_sub_u32) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_sub_i32) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<int, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_sub_u64) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_sub_f32) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<float, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_sub_f64) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_sub<double, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}
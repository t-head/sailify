#include "test_buf.h"

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_and(T *dst) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  T value = 1 << wid;
  if constexpr (sizeof(T) == 4) {
    value |= value << 16;
  } else {
    value |= value << 32;
  }
  if ((tid & 0x1f) == 0) {
    __nv_atomic_and(dst, ~value, order, scope);
  }
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_and() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_and with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  testDDst = testHDst.initialize([](int i) { return T(-1); });

  kernel_nv_atomic_and<T, order, scope><<<1, 128>>>(testDDst);
  testHDst = testDDst;

  T golden = 0;
  if constexpr (sizeof(T) == 4) {
    golden = 0xfff0fff0;
  } else {
    golden = 0xfffffff0fffffff0ul;
  }
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }

  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_and_b32) {
  bool ret = true;
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_and_b64) {
  bool ret = true;
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_fetch_and(T *dst, T *rtn) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  T value = 1 << wid;
  if constexpr (sizeof(T) == 4) {
    value |= value << 16;
  } else {
    value |= value << 32;
  }
  if ((tid & 0x1f) == 0) {
    rtn[wid] = __nv_atomic_fetch_and(dst, ~value, order, scope);
  }
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_fetch_and() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_fetch_and with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  auto testHRtn = testBuf<T, memory_scope::host>(4);
  auto testDRtn = testBuf<T, memory_scope::device>(4);
  testDDst = testHDst.initialize([](int i) { return T(-1); });

  kernel_nv_atomic_fetch_and<T, order, scope><<<1, 128>>>(testDDst, testDRtn);
  testHDst = testDDst;
  testHRtn = testDRtn;

  T golden = 0;
  if constexpr (sizeof(T) == 4) {
    golden = 0xfff0fff0;
  } else {
    golden = 0xfffffff0fffffff0ul;
  }
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }

  TLOG_INFO << "Checkin rtn values...\n";
  testHRtn.print(true);
  int offset = (sizeof(T) == 4) ? 16 : 32;
  uint32_t v2 = (sizeof(T) == 4) ? 0 : 0xffff0000;
  auto get_rtn = [&](uint32_t v) { return (T(v) << offset) | v; };
  bool check_rtn = (testHRtn[0] == get_rtn(v2 | 0xffff)) || (testHRtn[0] == get_rtn(v2 | 0xfffd)) ||
                   (testHRtn[0] == get_rtn(v2 | 0xfffb)) || (testHRtn[0] == get_rtn(v2 | 0xfff7)) ||
                   (testHRtn[0] == get_rtn(v2 | 0xfff9)) || (testHRtn[0] == get_rtn(v2 | 0xfff5)) ||
                   (testHRtn[0] == get_rtn(v2 | 0xfff3)) || (testHRtn[0] == get_rtn(v2 | 0xfff1));
  pass &= check_rtn;
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fetch_and_b32) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_and_b64) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_and<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}
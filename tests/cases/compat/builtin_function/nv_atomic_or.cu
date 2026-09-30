#include "test_buf.h"

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_or(T *dst) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  T value;
  if constexpr (sizeof(T) == 4) {
    value = ((wid + 1) << 16) | (wid + 1);
  } else {
    value = ((T(wid + 1) << 32) | (wid + 1));
  }
  __nv_atomic_or(dst, value, order, scope);
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_or() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_or with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  testDDst = testHDst.clear();

  kernel_nv_atomic_or<T, order, scope><<<1, 128>>>(testDDst);
  testHDst = testDDst;

  T golden = 0;
  if constexpr (sizeof(T) == 4) {
    golden = 0x70007;
  } else {
    golden = 0x700000007ul;
  }
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }

  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_or_b32) {
  bool ret = true;
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_or_b64) {
  bool ret = true;
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_fetch_or(T *dst, T *rtn) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  T value = 1 << wid;
  if constexpr (sizeof(T) == 4) {
    value |= value << 16;
  } else {
    value |= value << 32;
  }
  if ((tid & 0x1f) == 0) {
    rtn[wid] = __nv_atomic_fetch_or(dst, value, order, scope);
  }
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_fetch_or() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_fetch_or with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  auto testHDst = testBuf<T, memory_scope::host>(1);
  auto testDDst = testBuf<T, memory_scope::device>(1);
  auto testHRtn = testBuf<T, memory_scope::host>(4);
  auto testDRtn = testBuf<T, memory_scope::device>(4);
  testDDst = testHDst.clear();

  kernel_nv_atomic_fetch_or<T, order, scope><<<1, 128>>>(testDDst, testDRtn);
  testHDst = testDDst;
  testHRtn = testDRtn;

  T golden = 0;
  if constexpr (sizeof(T) == 4) {
    golden = 0xf000f;
  } else {
    golden = 0xf0000000ful;
  }
  bool pass = (golden == testHDst[0]);
  if (!pass) {
    TLOG_ERROR << "golden: 0x" << std::hex << golden << ", value: 0x"
      << testHDst[0] << "\n";
  }

  TLOG_INFO << "Checkin rtn values...\n";
  testHRtn.print(true);
  int offset = (sizeof(T) == 4) ? 16 : 32;
  auto get_rtn = [&](int v) { return (T(v) << offset) | v; };
  bool check_rtn = (testHRtn[0] == get_rtn(0)) || (testHRtn[0] == get_rtn(2)) ||
                   (testHRtn[0] == get_rtn(4)) || (testHRtn[0] == get_rtn(8)) ||
                   (testHRtn[0] == get_rtn(6)) || (testHRtn[0] == get_rtn(10)) ||
                   (testHRtn[0] == get_rtn(12)) || (testHRtn[0] == get_rtn(14));
  pass &= check_rtn;
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fetch_or_b32) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_fetch_or_b64) {
  bool ret = true;
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_CONSUME, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_fetch_or<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}
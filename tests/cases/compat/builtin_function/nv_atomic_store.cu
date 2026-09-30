#include "test_buf.h"

template<typename T, int order, int scope>
__global__ void kernel_nv_atomic_store(T *dst, T*src) {
  int tid = threadIdx.x;
  __nv_atomic_store(&dst[tid], &src[tid], order, scope);
}

template<typename T, int order, int scope>
inline bool test_nv_atomic_store() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_store with type: " << tname
    << ", order: " << order << ", scope: " << scope << "\n";
  int n = 128;
  auto testHSrc = testBuf<T, memory_scope::host>(n);
  auto testHDst = testBuf<T, memory_scope::host>(n);
  auto testDSrc = testBuf<T, memory_scope::device>(n);
  auto testDDst = testBuf<T, memory_scope::device>(n);
  testDSrc = testHSrc.initialize([](int i) {
    T ret;
    if constexpr (std::is_same_v<T, uint4>) {
      ret = make_uint4(i, i, i, i);
      return ret;
    } else {
      ret = 0;
      for (int l = 0; l < sizeof(T); l ++) {
        ret |= static_cast<T>(i) << (l * 8);
      }
      return ret;
    }
  });
  kernel_nv_atomic_store<T, order, scope><<<1, n>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  testHDst.foreach([&](T v, int i) {
    T golden = testHSrc[i];
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_store_b8) {
  bool ret = true;
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint8_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_store_b16) {
  bool ret = true;
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint16_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_store_b32) {
  bool ret = true;
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint32_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_store_b64) {
  bool ret = true;
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint64_t, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

TEST(builtin_func, nv_atomic_store_b128) {
  bool ret = true;
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_THREAD>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_BLOCK>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_DEVICE>();
  ret &= test_nv_atomic_store<uint4, __NV_ATOMIC_SEQ_CST, __NV_THREAD_SCOPE_SYSTEM>();
  EXPECT_TRUE(ret);
}

#include "test_buf.h"

template<int order>
__global__ void kernel_fence_thread(__uint128_t *dst, __uint128_t *src) {
  int tid = threadIdx.x;
  __uint128_t v1 = (__uint128_t)tid << 96 | (__uint128_t)tid << 64 | (__uint128_t)tid << 32 | tid;
  __uint128_t v2 = (__uint128_t)(tid + 100) << 96 | (__uint128_t)(tid + 100) << 64 |
                   (__uint128_t)(tid + 100) << 32 | (tid + 100);
  src[tid] = v1;
  __nv_atomic_thread_fence(order, __NV_THREAD_SCOPE_THREAD);
  src[tid + blockDim.x] = v2;
  dst[tid] = v2;
  dst[tid + blockDim.x] = v1;
}

template<int order>
inline bool test_nv_atomic_fence_thread() {
  TLOG_INFO << "Testing nv_atomic_thread_fence with scope __NV_THREAD_SCOPE_THREAD and order: "
    << order << "\n";
  int n = 128;
  auto testHSrc = testBuf<__uint128_t, memory_scope::host>(n * 2);
  auto testHDst = testBuf<__uint128_t, memory_scope::host>(n * 2);
  auto testDSrc = testBuf<__uint128_t, memory_scope::device>(n * 2);
  auto testDDst = testBuf<__uint128_t, memory_scope::device>(n * 2);

  kernel_fence_thread<order><<<1, n>>>(testDDst, testDSrc);
  testHSrc = testDSrc;
  testHDst = testDDst;

  bool pass = true;
  testHSrc.foreach([&](__uint128_t v, int i) {
    int section = i / n;
    int tid = i % n;
    __uint128_t golden = section ? (__uint128_t)(tid + 100) << 96 | (__uint128_t)(tid + 100) << 64 |
                   (__uint128_t)(tid + 100) << 32 | (tid + 100) : (__uint128_t)tid << 96 |
                   (__uint128_t)tid << 64 | (__uint128_t)tid << 32 | tid;
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "HSrc Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  testHDst.foreach([&](__uint128_t v, int i) {
    int section = i / n;
    int tid = i % n;
    __uint128_t golden = (section == 0) ? (__uint128_t)(tid + 100) << 96 | (__uint128_t)(tid + 100) << 64 |
                   (__uint128_t)(tid + 100) << 32 | (tid + 100) : (__uint128_t)tid << 96 |
                   (__uint128_t)tid << 64 | (__uint128_t)tid << 32 | tid;
    bool check = (golden == v);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "HDst Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fence_thread) {
  bool ret = true;
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_fence_thread<__NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<int order>
__global__ void kernel_fence_block(__uint128_t *dst, volatile __uint128_t *src) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lid = tid & 0x1f;
  int v1 = 29;
  __uint128_t golden = (__uint128_t)v1 << 96 | (__uint128_t)v1 << 64 | (__uint128_t)v1 << 32 | v1;
  __uint128_t v2 = (__uint128_t)lid << 96 | (__uint128_t)lid << 64 | (__uint128_t)lid << 32 | lid;
  if (wid == 31) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      __uint128_t v = (__uint128_t)l << 96 | (__uint128_t)l << 64 | (__uint128_t)l << 32 | l;
      src[lid] = v;
    }
    __nv_atomic_thread_fence(order, __NV_THREAD_SCOPE_BLOCK);
    src[lid + 32] = v2;
  } else if (wid == 0) {
    while (src[lid] != golden);
    __nv_atomic_thread_fence(__NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_BLOCK);
    dst[lid] = src[lid + 32];
  }
}

template<int order>
inline bool test_nv_atomic_fence_block() {
  TLOG_INFO << "Testing nv_atomic_thread_fence with scope __NV_THREAD_SCOPE_BLOCK and order: "
    << order << "\n";
  int nthreads = 1024;
  auto testHSrc = testBuf<__uint128_t, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<__uint128_t, memory_scope::host>(32);
  auto testDSrc = testBuf<__uint128_t, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<__uint128_t, memory_scope::device>(32);

  kernel_fence_block<order><<<1, nthreads>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  testHDst.foreach([&](__uint128_t v, int i) {
    __uint128_t golden = (__uint128_t)i << 96 | (__uint128_t)i << 64 | (__uint128_t)i << 32 | i;
    bool check = (v == golden);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "HDst Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fence_block) {
  bool ret = true;
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_fence_block<__NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<int order>
__global__ void kernel_fence_device(__uint128_t *dst, volatile __uint128_t *src) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = tid & 0x1f;
  int v1 = 29;
  __uint128_t golden = (__uint128_t)v1 << 96 | (__uint128_t)v1 << 64 | (__uint128_t)v1 << 32 | v1;
  __uint128_t v2 = (__uint128_t)lid << 96 | (__uint128_t)lid << 64 | (__uint128_t)lid << 32 | lid;
  if (bid == 1) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      __uint128_t v = (__uint128_t)l << 96 | (__uint128_t)l << 64 | (__uint128_t)l << 32 | l;
      src[lid] = v;
    }
    __nv_atomic_thread_fence(order, __NV_THREAD_SCOPE_DEVICE);
    src[lid + 32] = v2;
  } else if (bid == 0) {
    while (src[lid] != golden) {
      __nanosleep(100);
    };
    __nv_atomic_thread_fence(__NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
    dst[lid] = src[lid + 32];
  }
}

template<int order>
inline bool test_nv_atomic_fence_device() {
  TLOG_INFO << "Testing nv_atomic_thread_fence with scope __NV_THREAD_SCOPE_DEVICE and order: "
    << order << "\n";
  auto testHSrc = testBuf<__uint128_t, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<__uint128_t, memory_scope::host>(32);
  auto testDSrc = testBuf<__uint128_t, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<__uint128_t, memory_scope::device>(32);

  kernel_fence_device<order><<<2, 32>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  testHDst.foreach([&](__uint128_t v, int i) {
    __uint128_t golden = (__uint128_t)i << 96 | (__uint128_t)i << 64 | (__uint128_t)i << 32 | i;
    bool check = (v == golden);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "HDst Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fence_device) {
  bool ret = true;
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_fence_device<__NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<int order>
__global__ void kernel_fence_system(__uint128_t *dst, volatile __uint128_t *src) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = tid & 0x1f;
  int v1 = 29;
  __uint128_t golden = (__uint128_t)v1 << 96 | (__uint128_t)v1 << 64 | (__uint128_t)v1 << 32 | v1;
  __uint128_t v2 = (__uint128_t)lid << 96 | (__uint128_t)lid << 64 | (__uint128_t)lid << 32 | lid;
  if (bid == 1) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      __uint128_t v = (__uint128_t)l << 96 | (__uint128_t)l << 64 | (__uint128_t)l << 32 | l;
      src[lid] = v;
    }
    __nv_atomic_thread_fence(order, __NV_THREAD_SCOPE_SYSTEM);
    src[lid + 32] = v2;
  } else if (bid == 0) {
    while (src[lid] != golden) {
      __nanosleep(100);
    };
    __nv_atomic_thread_fence(__NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM);
    dst[lid] = src[lid + 32];
  }
}

template<int order>
inline bool test_nv_atomic_fence_system() {
  TLOG_INFO << "Testing nv_atomic_thread_fence with scope __NV_THREAD_SCOPE_SYSTEM and order: "
    << order << "\n";
  auto testHSrc = testBuf<__uint128_t, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<__uint128_t, memory_scope::host>(32);
  auto testDSrc = testBuf<__uint128_t, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<__uint128_t, memory_scope::device>(32);

  kernel_fence_system<order><<<2, 32>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  testHDst.foreach([&](__uint128_t v, int i) {
    __uint128_t golden = (__uint128_t)i << 96 | (__uint128_t)i << 64 | (__uint128_t)i << 32 | i;
    bool check = (v == golden);
    pass &= check;
    if (!check) {
      TLOG_ERROR << "HDst Idx: " << i << ", golden: 0x" << std::hex
        << golden << ", value: 0x" << v << std::dec << "\n";
    }
  });
  TLOG_INFO << "Test result: " << (pass ? "pass !" : "fail !") << "\n";
  return pass;
}

TEST(builtin_func, nv_atomic_fence_system) {
  bool ret = true;
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_fence_system<__NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}
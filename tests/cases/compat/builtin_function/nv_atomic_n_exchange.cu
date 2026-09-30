#include "test_buf.h"

template<typename T, int order>
__global__ void kernel_exchange_n_thread(T *dst, T *src) {
  int tid = threadIdx.x;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  dst[tid] = gen_val(tid);
  T val = gen_val(tid + 10);
  __threadfence();
  T reg = 0;
  if constexpr (sizeof(T) == 4) {
    reg =  __nv_atomic_exchange_n((uint32_t*)&dst[tid], val,
      order,__NV_THREAD_SCOPE_THREAD);
  } else {
    reg = __nv_atomic_exchange_n((uint64_t*)&dst[tid], val, order,__NV_THREAD_SCOPE_THREAD);
  }
}

template<typename T, int order>
inline bool test_nv_atomic_exchange_n_thread() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_exchange_n with type: " << tname
    << ", order: " << order << ", scope: __NV_THREAD_SCOPE_THREAD" << "\n";
  int n = 128;
  auto testHSrc = testBuf<T, memory_scope::host>(n);
  auto testHDst = testBuf<T, memory_scope::host>(n);
  auto testDSrc = testBuf<T, memory_scope::device>(n);
  auto testDDst = testBuf<T, memory_scope::device>(n);

  kernel_exchange_n_thread<T, order><<<1, n>>>(testDDst, testDSrc);
  testHDst = testDDst;
  testHSrc = testDSrc;

  bool pass = true;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  testHDst.foreach([&](T v, int i) {
    T golden = gen_val(i + 10);
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

TEST(builtin_func, nv_atomic_exchange_n_thread) {
  bool ret = true;
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_thread<uint32_t, __NV_ATOMIC_SEQ_CST>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_thread<uint64_t, __NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<typename T, int order>
__global__ void kernel_exchange_n_block(T *dst, T *src) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lid = tid & 0x1f;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  if (wid == 31) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      src[lid] = gen_val(l);
    }
    T reg = 0;
    T val = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      reg = __nv_atomic_exchange_n((uint32_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_BLOCK);
    } else {
      reg = __nv_atomic_exchange_n((uint64_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_BLOCK);
    }
  } else if (wid == 0) {
    T expected = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      while(!__nv_atomic_compare_exchange((uint32_t*)&src[lid + 32], (uint32_t*)&expected,
        (uint32_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_BLOCK)) {
        expected = gen_val(100);
      }
    } else {
      while(!__nv_atomic_compare_exchange((uint64_t*)&src[lid + 32], (uint64_t*)&expected,
        (uint64_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_BLOCK)) {
        expected = gen_val(100);
      }
    }
    dst[lid] = src[lid];
  }
}

template<typename T, int order>
inline bool test_nv_atomic_exchange_n_block() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_exchange_n with type: " << tname
    << ", order: " << order << ", scope: __NV_THREAD_SCOPE_BLOCK" << "\n";
  int n = 1024;
  auto testHSrc = testBuf<T, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<T, memory_scope::host>(32);
  auto testDSrc = testBuf<T, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<T, memory_scope::device>(32);

  kernel_exchange_n_block<T, order><<<1, n>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  testHDst.foreach([&](T v, int i) {
    T golden = gen_val(29);
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

TEST(builtin_func, nv_atomic_exchange_n_block) {
  bool ret = true;
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_block<uint32_t, __NV_ATOMIC_SEQ_CST>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_block<uint64_t, __NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<typename T, int order>
__global__ void kernel_exchange_n_device(T *dst, T *src) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = tid & 0x1f;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  if (bid == 1) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      src[lid] = gen_val(l);
    }
    T reg = 0;
    T val = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      reg = __nv_atomic_exchange_n((uint32_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_DEVICE);
    } else {
      reg = __nv_atomic_exchange_n((uint64_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_DEVICE);
    }
  } else if (bid == 0) {
    T expected = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      while(!__nv_atomic_compare_exchange((uint32_t*)&src[lid + 32], (uint32_t*)&expected,
        (uint32_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_DEVICE)) {
        expected = gen_val(100);
      }
    } else {
      while(!__nv_atomic_compare_exchange((uint64_t*)&src[lid + 32], (uint64_t*)&expected,
        (uint64_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_DEVICE)) {
        expected = gen_val(100);
      }
    }
    dst[lid] = src[lid];
  }
}

template<typename T, int order>
inline bool test_nv_atomic_exchange_n_device() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_exchange_n with type: " << tname
    << ", order: " << order << ", scope: __NV_THREAD_SCOPE_DEVICE" << "\n";
  auto testHSrc = testBuf<T, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<T, memory_scope::host>(32);
  auto testDSrc = testBuf<T, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<T, memory_scope::device>(32);

  kernel_exchange_n_device<T, order><<<2, 32>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  testHDst.foreach([&](T v, int i) {
    T golden = gen_val(29);
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

TEST(builtin_func, nv_atomic_exchange_n_device) {
  bool ret = true;
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_device<uint32_t, __NV_ATOMIC_SEQ_CST>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_device<uint64_t, __NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}

template<typename T, int order>
__global__ void kernel_exchange_n_system(T *dst, T *src) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = tid & 0x1f;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  if (bid == 1) {
    #pragma unroll
    for (int l = 0; l < 30; l ++) {
      src[lid] = gen_val(l);
    }
    T reg = 0;
    T val = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      reg = __nv_atomic_exchange_n((uint32_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_SYSTEM);
    } else {
      reg = __nv_atomic_exchange_n((uint64_t*)&src[lid + 32], val,
        order,__NV_THREAD_SCOPE_SYSTEM);
    }
  } else if (bid == 0) {
    T expected = gen_val(100);
    if constexpr (sizeof(T) == 4) {
      while(!__nv_atomic_compare_exchange((uint32_t*)&src[lid + 32], (uint32_t*)&expected,
        (uint32_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_SYSTEM)) {
        expected = gen_val(100);
      }
    } else {
      while(!__nv_atomic_compare_exchange((uint64_t*)&src[lid + 32], (uint64_t*)&expected,
        (uint64_t*)&expected, false, __NV_ATOMIC_ACQUIRE, order, __NV_THREAD_SCOPE_SYSTEM)) {
        expected = gen_val(100);
      }
    }
    dst[lid] = src[lid];
  }
}

template<typename T, int order>
inline bool test_nv_atomic_exchange_n_system() {
  std::string tname = print_types<T>();
  TLOG_INFO << "Testing nv_atomic_exchange_n with type: " << tname
    << ", order: " << order << ", scope: __NV_THREAD_SCOPE_SYSTEM" << "\n";
  auto testHSrc = testBuf<T, memory_scope::host>(32 * 2);
  auto testHDst = testBuf<T, memory_scope::host>(32);
  auto testDSrc = testBuf<T, memory_scope::device>(32 * 2);
  auto testDDst = testBuf<T, memory_scope::device>(32);

  kernel_exchange_n_system<T, order><<<2, 32>>>(testDDst, testDSrc);
  testHDst = testDDst;

  bool pass = true;
  auto gen_val = [](int id) {
    if constexpr (sizeof(T) == 4) {
      return ((id + 1) << 16) | (id + 1);
    } else {
      return (T(id + 1) << 32) | (id + 1);
    }
  };
  testHDst.foreach([&](T v, int i) {
    T golden = gen_val(29);
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

TEST(builtin_func, nv_atomic_exchange_n_system) {
  bool ret = true;
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_system<uint32_t, __NV_ATOMIC_SEQ_CST>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_RELAXED>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_CONSUME>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_ACQUIRE>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_RELEASE>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_ACQ_REL>();
  ret &= test_nv_atomic_exchange_n_system<uint64_t, __NV_ATOMIC_SEQ_CST>();
  EXPECT_TRUE(ret);
}
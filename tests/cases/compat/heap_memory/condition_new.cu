#include "test_buf.h"

__device__ int pcnt;

__global__ void kernel_cond_malloc1(int *dinput, int *ptarr[]) {
  int tid = threadIdx.x;
  if (tid == 0) {
    pcnt = 0;
  }
  __syncthreads();
  if (dinput[tid] % 2 == 0) {
    ptarr[atomicAdd(&pcnt, 1)] = new int{1};
  }
}

__global__ void kernel_cond_malloc2(int *dinput, int *ptarr[]) {
  int tid = threadIdx.x;
  if (tid <= 24 && (__ballot_sync(0xffffffff, dinput[tid] % 3 == 0) & (1u << tid))) {
    ptarr[atomicAdd(&pcnt, 1)] = new int{2};
  }
}

__global__ void kernel_use_and_free(int *dout, int *ptarr[]) {
  int tid = threadIdx.x;
  if (tid >= pcnt) {
    dout[tid] = -1;
    return;
  }
  dout[tid] = *ptarr[tid];
  delete ptarr[tid];
}

TEST(heap_memory, condition_new) {
  auto testDInput = testBuf<int, memory_scope::device>(32);
  auto testDOut = testBuf<int, memory_scope::device>(32);
  auto testHOut = testBuf<int, memory_scope::host>(32);
  auto testDPTR = testBuf<int*, memory_scope::device>(32);
  testDInput = {0, 2, 3, 5, 4, 6, 7, 9, 11, 10, 15, 17, 1, 16,
    19, 21, 5, 5, 5, 5, 5, 9, 11, 11, 11, 9, 9, 9, 9, 9, 9, 5};
  kernel_cond_malloc1<<<1, 32>>>(testDInput, testDPTR);
  kernel_cond_malloc2<<<1, 32>>>(testDInput, testDPTR);
  kernel_use_and_free<< <1, 32>> >(testDOut, testDPTR);
  testHOut = testDOut;
  bool pass = true;
  testHOut.foreach([&](int v, int i) {
    int golden = (i < 6) ? 1 : (i < 13) ? 2 : -1;
    bool ret = (v == golden);
    pass &= ret;
    if (!ret) {
      TLOG_ERROR << "Expected value: " << golden << ", actual: " << v
        << ", index: " << i << "\n";
    }
  });
  TLOG_INFO << "Running " << (pass ? "pass !" : "failed !") << "\n";
  EXPECT_TRUE(pass);
}
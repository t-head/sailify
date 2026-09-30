#include "test_buf.h"

static __global__ void nv_bswap(uint16_t *dp16, uint32_t *dp32, uint64_t *dp64) {
  int tid = threadIdx.x;
  dp16[tid] = __nv_bswap16(dp16[tid]);
  dp32[tid] = __nv_bswap32(dp32[tid]);
  dp64[tid] = __nv_bswap64(dp64[tid]);
}

TEST(builtin_func, nv_bswap_cuda_12_8) {
  int n = 3;
  auto testHBuf16 = testBuf<uint16_t, memory_scope::host>(n);
  auto testHBuf32 = testBuf<uint32_t, memory_scope::host>(n);
  auto testHBuf64 = testBuf<uint64_t, memory_scope::host>(n);
  auto testHRBuf16 = testBuf<uint16_t, memory_scope::host>(n);
  auto testHRBuf32 = testBuf<uint32_t, memory_scope::host>(n);
  auto testHRBuf64 = testBuf<uint64_t, memory_scope::host>(n);
  testHBuf16 = std::vector<uint16_t>{0xabcd, 0x5678, 0x1231};
  testHBuf32 = std::vector<uint32_t>{0xabcd5678, 0x1234abcd, 0xeeffffee};
  testHBuf64 = std::vector<uint64_t>{0xabcd567812345678ul, 0x1234abcdeeff7788ul, 0xeeffffeeaabbbbaaul};
  auto testDBuf16 = testBuf<uint16_t, memory_scope::device>(n);
  auto testDBuf32 = testBuf<uint32_t, memory_scope::device>(n);
  auto testDBuf64 = testBuf<uint64_t, memory_scope::device>(n);
  testDBuf16 = testHBuf16;
  testDBuf32 = testHBuf32;
  testDBuf64 = testHBuf64;
  nv_bswap<<<1, 3>>>(testDBuf16, testDBuf32, testDBuf64);
  testHRBuf16 = testDBuf16;
  testHRBuf32 = testDBuf32;
  testHRBuf64 = testDBuf64;
  TLOG_INFO << "Verifying device bswap result...\n";
  bool pass = true;
  testHRBuf16.foreach([&](uint16_t v, int i) {
    uint16_t goldens[] = {0xcdab, 0x7856, 0x3112};
    if (v != goldens[i]) {
      TLOG_INFO << std::hex << "HBuf16 Expected value: 0x" << goldens[i]
        << ", actual value: 0x" << v << std::dec << "\n";
      pass = false;
    }
  });
  testHRBuf32.foreach([&](uint32_t v, int i) {
    uint32_t goldens[] = {0x7856cdab, 0xcdab3412, 0xeeffffee};
    if (v != goldens[i]) {
      TLOG_INFO << std::hex << "HBuf32 Expected value: 0x" << goldens[i]
        << ", actual value: 0x" << v << std::dec << "\n";
      pass = false;
    }
  });
  testHRBuf64.foreach([&](uint64_t v, int i) {
    uint64_t goldens[] = {0x785634127856cdabul, 0x8877ffeecdab3412ul, 0xaabbbbaaeeffffeeul};
    if (v != goldens[i]) {
      TLOG_INFO << std::hex << "HBuf64 Expected value: 0x" << goldens[i]
        << ", actual value: 0x" << v << std::dec << "\n";
      pass = false;
    }
  });
  TLOG_INFO << "Result " << (pass ? "pass !" : "fail !") << "\n";

  TLOG_INFO << "Verifying host bswap result...\n";
  uint16_t h1 = __nv_bswap16(0xcdab);
  uint32_t h2 = __nv_bswap32(0x5678abcd);
  uint64_t h3 = __nv_bswap64(0x12345678abcd5678ul);
  TLOG_INFO << std::hex << "h1 : 0x" << h1 << ", h2: 0x" << h2 << ", h3: 0x" << h3 << "\n";
  pass &= ((h1 == 0xabcd) & (h2 == 0xcdab7856) & (h3 == 0x7856cdab78563412ul));
  EXPECT_TRUE(pass);
}

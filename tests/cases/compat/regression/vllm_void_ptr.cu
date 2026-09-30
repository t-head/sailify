#include "test_buf.h"

struct __align__(16) RankData { const void *__restrict__ ptrs[8]; };

class CustomAllreduce {
public:
  RankData *d_rank_data_base_, *d_rank_data_end_;

  __device__ CustomAllreduce(void *rank_data, size_t rank_data_sz) : d_rank_data_base_(reinterpret_cast<RankData *>(rank_data)),
    d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {}
};

__global__ void kernel(unsigned long long* dp) {
    CustomAllreduce CAR(dp, 1);
    dp[1] = reinterpret_cast<unsigned long long>(CAR.d_rank_data_end_);
}

TEST(regression, vllm_void_ptr) {
  EXPECT_TRUE(true);
}
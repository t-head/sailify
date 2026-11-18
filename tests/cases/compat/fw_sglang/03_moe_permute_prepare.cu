#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>


namespace ffi = tvm::ffi;
#define CHECK_CPU(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCPU) << #x " must be a host tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TVM_FFI_ICHECK_EQ(x.stride(-1), 1) \
  #x "must be contiguous at last dimension";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_CUDA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";
#define CHECK_DEVICE(a, b)                                           \
  TVM_FFI_ICHECK_EQ(a.device().device_type, b.device().device_type); \
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);


#include <algorithm>
#include <cuda_runtime.h>
#include <limits>

namespace sglang {


__device__ __forceinline__ int32_t lower_bound(const int32_t* __restrict__ data, int32_t n, int32_t target) {
  int32_t lo = 0, hi = n;
  while (lo < hi) {
    int32_t mid = lo + (hi - lo) / 2;
    if (data[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}


__global__ void moe_permute_prepare_kernel(
    const int32_t* __restrict__ sorted_topk_ids,
    const int64_t* __restrict__ reorder_ids,
    void* __restrict__ expert_offsets,
    int32_t* __restrict__ src2dst,
    int32_t num_experts,
    int32_t numel,
    bool use_int64_offset,
    bool is_ep) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int32_t neg_count = 0;
  if (is_ep) neg_count = lower_bound(sorted_topk_ids, numel, 0);

  for (int e = tid; e <= num_experts; e += stride) {
    int32_t offset;
    if (e < num_experts) {
      offset = lower_bound(sorted_topk_ids, numel, e) - neg_count;
    } else {
      offset = numel - neg_count;
    }

    if (use_int64_offset) {
      reinterpret_cast<int64_t*>(expert_offsets)[e] = static_cast<int64_t>(offset);
    } else {
      reinterpret_cast<int32_t*>(expert_offsets)[e] = offset;
    }
  }

  for (int i = tid; i < numel; i += stride) {
    src2dst[reorder_ids[i]] = i - neg_count;
  }
}



}

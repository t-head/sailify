#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define VEC_SIZE 4
namespace sglang {

using Vec = int4;

inline uint32_t next_pow2(uint32_t x) noexcept {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

namespace moe_lora_merged {

__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}

template <typename scalar_t>
__device__ __forceinline__ int compute_virtual_id(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    size_t i,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool compact) {
  int m = static_cast<int>(i) / top_k;
  int lora_id = token_lora_mapping[m];
  bool mask_val = lora_id >= 0;
  int safe_lora = lora_id > 0 ? lora_id : 0;

  int base = shared_outer ? 0 : static_cast<int>(topk_ids[i]);
  if (ep_local) {
    bool owned = base >= local_expert_offset && base < local_expert_offset + local_num_experts;
    base = owned ? base : -1;
  }
  if (!mask_val || base < 0) return -1;
  if (compact) return base - local_expert_offset;
  return base + safe_lora * num_experts_for_weight;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    if (do_skip && vid < 0) continue;
    int32_t expert_id = vid + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    bool* __restrict__ token_lora_mask,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;
  int32_t* prefix = shared_counts + num_experts;
  int32_t* scan_buf = prefix + num_experts + 1;
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    if (!do_skip || vid >= 0) {
      atomicAdd(&shared_counts[vid + 1], 1);
    }
    if (static_cast<int>(i) % top_k == 0) {
      int m = static_cast<int>(i) / top_k;
      token_lora_mask[m] = token_lora_mapping[m] >= 0;
    }
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  int32_t* warp_sums = scan_buf + scan_size;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  if (tid < num_experts) prefix[tid] = scan_buf[tid];

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2 + (compact ? local_expert_offset : 0);
  }
}

template <typename scalar_t>
__global__ void fused_align_scatter_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    bool* __restrict__ token_lora_mask,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    const int32_t scan_size,
    int32_t max_num_tokens_padded,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;
  int32_t* prefix = shared_counts + num_experts;
  int32_t* scan_buf = prefix + num_experts + 1;
  int32_t* warp_sums = scan_buf + scan_size;
  int32_t* cursor = warp_sums + WARP_SIZE;
  int32_t* svids = cursor + num_experts;
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;

  {
    Vec fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
    int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      out_ptr[i] = fill_vec;
    }
  }
  if (tid < num_experts) shared_counts[tid] = 0;
  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    svids[i] = vid;
    if (!do_skip || vid >= 0) {
      atomicAdd(&shared_counts[vid + 1], 1);
    }
    if (static_cast<int>(i) % top_k == 0) {
      int m = static_cast<int>(i) / top_k;
      token_lora_mask[m] = token_lora_mapping[m] >= 0;
    }
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();
  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();
  int off = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + off;
  __syncthreads();
  if (tid < num_experts) prefix[tid] = scan_buf[tid];
  if (tid <= num_experts) cumsum[tid] = prefix[tid];
  __syncthreads();

  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2 + (compact ? local_expert_offset : 0);
  }
  if (tid < num_experts) cursor[tid] = prefix[tid];
  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int vid = svids[i];
    if (do_skip && vid < 0) continue;
    int bucket = vid + 1;
    int pos = atomicAdd(&cursor[bucket], 1);
    sorted_token_ids[pos] = i;
  }
}

}

}

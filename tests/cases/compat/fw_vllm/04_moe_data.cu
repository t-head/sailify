#include <cudaTypedefs.h>
#include <iostream>

constexpr uint64_t THREADS_PER_EXPERT = 512;
                                                                       
constexpr int SWAP_AB_THRESHOLD = 64;

template <bool SWAP_AB>
__global__ void compute_problem_sizes(const int32_t* __restrict__ topk_ids,
                                      int32_t* problem_sizes1,
                                      int32_t* problem_sizes2,
                                      int32_t* atomic_buffer,
                                      const int topk_length, const int n,
                                      const int k, const bool is_gated) {
  int expert_id = blockIdx.x;
                                                                 
                                                                 
  int const n1 = is_gated ? 2 * n : n;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    if constexpr (!SWAP_AB) {
      problem_sizes1[expert_id * 3] = final_occurrences;
      problem_sizes1[expert_id * 3 + 1] = n1;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = final_occurrences;
      problem_sizes2[expert_id * 3 + 1] = k;
      problem_sizes2[expert_id * 3 + 2] = n;
    } else {
      problem_sizes1[expert_id * 3] = n1;
      problem_sizes1[expert_id * 3 + 1] = final_occurrences;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = k;
      problem_sizes2[expert_id * 3 + 1] = final_occurrences;
      problem_sizes2[expert_id * 3 + 2] = n;
    }
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts, const bool swap_ab) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* blockscale_offsets, int32_t* atomic_buffer, const int num_experts,
    const bool swap_ab) {
  int32_t tot_offset = 0;
  int32_t tot_offset_round = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    int32_t cur_offset =
        swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
    atomic_buffer[i] = tot_offset;
    tot_offset += cur_offset;
    expert_offsets[i + 1] = tot_offset;
    tot_offset_round += (cur_offset + (128 - 1)) / 128 * 128;
    blockscale_offsets[i + 1] = tot_offset_round;
  }
}

__global__ void compute_arg_sorts(const int32_t* __restrict__ topk_ids,
                                  const int32_t* __restrict__ expert_offsets,
                                  int32_t* input_permutation,
                                  int32_t* output_permutation,
                                  int32_t* atomic_buffer, const int topk_length,
                                  const int topk) {
  int const blk_expert_id = blockIdx.x;
  int const num_experts = gridDim.x;
  int32_t const num_tokens = expert_offsets[num_experts];

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    int const expert_id = topk_ids[i];
    if (expert_id == -1 && blockIdx.x == 0) {
                                                                      
                                                                       
                                                                           
                                                                                
                                                                         
                                
      output_permutation[i] = num_tokens;
    } else if (expert_id == blk_expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

namespace {}
template <bool SWAP_AB>
__global__ void compute_problem_sizes_from_expert_offsets(
    const int64_t* __restrict__ expert_first_token_offset,
    int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ problem_sizes2,
    const int num_experts, const int n, const int k) {
  int const expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) {
    return;
  }

  int64_t const m64 = expert_first_token_offset[expert_id + 1] -
                      expert_first_token_offset[expert_id];
  int32_t const m = static_cast<int32_t>(m64);

  int32_t* ps1 = problem_sizes1 + expert_id * 3;
  int32_t* ps2 = problem_sizes2 + expert_id * 3;

  if constexpr (!SWAP_AB) {
                  
    ps1[0] = m;
    ps1[1] = 2 * n;
    ps1[2] = k;
                
    ps2[0] = m;
    ps2[1] = k;
    ps2[2] = n;
  } else {
                                            
                  
    ps1[0] = 2 * n;
    ps1[1] = m;
    ps1[2] = k;
                
    ps2[0] = k;
    ps2[1] = m;
    ps2[2] = n;
  }
}

template <bool SWAP_AB>
__global__ void compute_batched_moe_data(
    int32_t* expert_offsets, int32_t* problem_sizes1, int32_t* problem_sizes2,
    const int32_t* __restrict__ expert_num_tokens, const int padded_m,
    const int n, const int k) {
  int expert_idx = threadIdx.x;
  expert_offsets[expert_idx] = expert_idx * padded_m;

  if constexpr (!SWAP_AB) {
    problem_sizes1[expert_idx * 3] = expert_num_tokens[expert_idx];
    problem_sizes1[expert_idx * 3 + 1] = 2 * n;
    problem_sizes1[expert_idx * 3 + 2] = k;
    problem_sizes2[expert_idx * 3] = expert_num_tokens[expert_idx];
    problem_sizes2[expert_idx * 3 + 1] = k;
    problem_sizes2[expert_idx * 3 + 2] = n;
  } else {
    problem_sizes1[expert_idx * 3] = 2 * n;
    problem_sizes1[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
    problem_sizes1[expert_idx * 3 + 2] = k;
    problem_sizes2[expert_idx * 3] = k;
    problem_sizes2[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
    problem_sizes2[expert_idx * 3 + 2] = n;
  }
}

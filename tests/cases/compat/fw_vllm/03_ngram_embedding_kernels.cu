


#pragma once


#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>


inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;


#pragma once


#include <optional>
#include <string>
#include <vector>


#ifndef USE_ROCM
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);


bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);


#endif


#ifndef USE_ROCM
#endif


#ifdef VLLM_ENABLE_FUSED_KDA_DECODE

#endif

#ifdef VLLM_ENABLE_FUSED_GDN_DECODE

#endif

#ifdef VLLM_ENABLE_KIMI_K3_ATTN_RES
#endif


#ifdef VLLM_ENABLE_COOPERATIVE_TOPK
#endif


using fptr_t = int64_t;
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
void free_shared_buffer(int64_t buffer);


                                     


#include <cstdint>

namespace vllm::ngram_embedding {

constexpr int kBlockThreads = 256;

__global__ void ComputeNGramIdsKernel(
    int batch_size, int ne_n, int ne_k,
    int* ne_weights,                       
    int* ne_mods,                          
    int* exclusive_ne_embedder_size_sums,  
    int* exclusive_req_len_sums,           
    int* ne_token_table,  
    int max_context_len,
    const int64_t* __restrict__ row_indices,  
    int* column_starts,                       
    int* n_gram_ids                           
) {
  const int req_id = blockIdx.x % batch_size;
  const int config_id = (blockIdx.x - req_id) / batch_size;
  
  
  const int k = config_id % ne_k;
  const int n = (config_id - config_id % ne_k) / ne_k;
  const int ne_weight_base_idx = n * ne_k * ne_n + k * ne_n;
  const int ne_mod = ne_mods[n * ne_k + k];
  for (int i = exclusive_req_len_sums[req_id] + threadIdx.x;
       i < exclusive_req_len_sums[req_id + 1]; i += blockDim.x) {
    uint64_t n_gram_id = 0;
    const int64_t current_token_offset = i - exclusive_req_len_sums[req_id];
    const int64_t req_token_table_index =
        row_indices[req_id] * static_cast<int64_t>(max_context_len);
    const int64_t current_token_table_index =
        req_token_table_index + column_starts[req_id] + current_token_offset;
    for (int j = 0; j < n + 2; j++) {
      if (current_token_table_index - j < req_token_table_index) {
        break;  
      }
      if (ne_token_table[current_token_table_index - j] < 0) {
        break;  
      }
      const uint64_t term =
          (uint64_t)ne_token_table[current_token_table_index - j] *
          (uint64_t)ne_weights[ne_weight_base_idx + j];
      n_gram_id += term % ne_mod;
    }
    n_gram_id %= ne_mod;
    n_gram_id += exclusive_ne_embedder_size_sums[n * ne_k + k];
    n_gram_ids[i * (ne_n - 1) * ne_k + n * ne_k + k] = (int)(n_gram_id);
  }
}

}  


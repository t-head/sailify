#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

                                

namespace {

                                             
                                        
template <typename T>
__global__ void max_pooling_1d_varlen_kernel(
    const T* input,
    T* output,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    const int* cache_lens,
    int batch_size,
    int num_heads,
    int max_seqlen_k,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks) {
  const int bidh = blockIdx.y;                      
  const int bidq_global = blockIdx.x;                                          

  int batch_idx = 0;
  int q_start = 0, q_end = 0, k_start = 0, k_end = 0;
  for (int b = 0; b < batch_size; b++) {
    q_start = cu_seqlens_q[b];
    q_end = cu_seqlens_q[b + 1];
    k_start = cu_seqlens_k[b];
    k_end = cu_seqlens_k[b + 1];
    if (bidq_global >= q_start && bidq_global < q_end) {
      batch_idx = b;
      break;
    }
  }

  const int bidq_local = bidq_global - q_start;
  const int seqlen_q = q_end - q_start;
  const int seqlen_k = k_end - k_start;
  if (bidq_local >= seqlen_q) return;

  const size_t total_q_all = static_cast<size_t>(cu_seqlens_q[batch_size]);
  const size_t in_offset =
      static_cast<size_t>(bidh) * total_q_all * max_seqlen_k + static_cast<size_t>(bidq_global) * max_seqlen_k;
  const T* in = input + in_offset;
  const size_t out_offset =
      static_cast<size_t>(bidh) * total_q_all * out_len + static_cast<size_t>(bidq_global) * out_len;
  T* out = output + out_offset;

  const int cache_len = cache_lens[batch_idx];
  const int off_bq = (bidq_local + cache_len) / block_size;
  const T pos_inf = static_cast<T>(static_cast<float>(INFINITY));

  for (int k = threadIdx.x; k < out_len; k += blockDim.x) {
    const int off_bk = k;
    const bool should_mask_inf = (off_bk < init_blocks) || ((off_bq >= off_bk) && (off_bq <= off_bk + local_blocks));

    if (should_mask_inf) {
      out[k] = pos_inf;
    } else {
      int start = k * stride - padding;
      int end = start + kernel_size;
      start = max(start, 0);
      end = min(end, seqlen_k);

      float max_val = -INFINITY;
      for (int i = start; i < end; i++) {
        const float v = static_cast<float>(in[i]);
        if (v > max_val) max_val = v;
      }
      out[k] = static_cast<T>(max_val);
    }
  }
}

}
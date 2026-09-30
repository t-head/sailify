#include <cuda_runtime.h>
#include <algorithm>
#include <optional>

                                                  

                                                           
                                                                 
inline __device__ float to_float(float u) {
  return u;
}
inline __device__ float to_float(half u) {
  return __half2float(u);
}
inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}
inline __device__ void from_float(float& d, float s) {
  d = s;
}
inline __device__ void from_float(half& d, float s) {
  d = __float2half(s);
}
inline __device__ void from_float(__nv_bfloat16& d, float s) {
  d = __float2bfloat16(s);
}

                                                                 
template <typename scalar_t, const uint NUM_THREADS>
__global__ void merge_attn_states_kernel(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    const uint num_tokens,
    const uint num_heads,
    const uint head_size) {
  using pack_128b_t = uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

                                                  
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;                   
  const uint head_offset = token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;

                                                                 
                                                                 
  float p_lse = prefix_lse[token_idx * num_heads + head_idx];
  float s_lse = suffix_lse[token_idx * num_heads + head_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
                     
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
                                                         
                                                  
      const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
                                                                 
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
                                                  
      from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

                        
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
  }
                                                       
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[token_idx * num_heads + head_idx] = out_lse;
  }
}
#define LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS)                          \
  {                                                                              \
    merge_attn_states_kernel<scalar_t, NUM_THREADS><<<grid, block, 0, stream>>>( \
        reinterpret_cast<scalar_t*>(output.data_ptr()),                          \
        reinterpret_cast<float*>(output_lse.data_ptr()),                         \
        reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),                   \
        reinterpret_cast<float*>(prefix_lse.data_ptr()),                         \
        reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),                   \
        reinterpret_cast<float*>(suffix_lse.data_ptr()),                         \
        num_tokens,                                                              \
        num_heads,                                                               \
        head_size);                                                              \
  }
#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                                 \
  {                                                                               \
    merge_attn_states_launcher<scalar_t>(v_a, s_a, v_b, s_b, v_merged, s_merged); \
  }

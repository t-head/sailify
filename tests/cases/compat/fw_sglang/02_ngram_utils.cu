#include <cuda_runtime.h>
#ifndef USE_ROCM
#else
#endif

                                                      
                         
                                    
                                       
                                            
                                              
__global__ void reconstructIndicesFromTreeMask(
    bool* tree_mask,
    int64_t* verified_seq_len,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int batch_size,
    int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= batch_size || tid >= draft_token_num) {
    return;
  }
  int base_offset = draft_token_num * draft_token_num;
                                                                    
  int token_idx = bid * draft_token_num;
                                                                
  int tree_mask_offset = bid * base_offset;

  int depth = 0;
  int parent_idx = -1;

  for (int i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
    if (tree_mask[start_idx + i]) {
      depth++;
      if (parent_idx == -1) {
        parent_idx = i;
      }
    }
  }
  retrive_index[token_idx + tid] = token_idx + tid;
  positions[token_idx + tid] = depth + verified_seq_len[bid];

  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  int next_sibling_idx = -1;
  if (parent_idx != -1) {
    for (int i = tid + 1; i < draft_token_num; i++) {
      int start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
      if (tree_mask[start_idx]) {
        bool is_sibling = true;
        int end_idx = tree_mask_offset + i * draft_token_num + i;
        for (int j = start_idx + 1; j < end_idx; ++j) {
          if (tree_mask[j]) {
            is_sibling = false;
            break;
          }
        }
        if (is_sibling) {
          next_sibling_idx = i;
          break;
        }
      }
    }
  }
  retrive_next_sibling[token_idx + tid] = next_sibling_idx;
}

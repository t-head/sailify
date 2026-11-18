#pragma once
#include <cuda.h>
#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)         \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                                 \
    get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float, LayoutSFA, LayoutSFB, ScaleConfig> \
        <<<1, num_experts, 0, stream>>>(                                                           \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                                      \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),                               \
            static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),                               \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                                            \
            static_cast<float**>(a_scales_ptrs.data_ptr()),                                        \
            static_cast<float**>(b_scales_ptrs.data_ptr()),                                        \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),                             \
            static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),                             \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                                          \
            static_cast<float*>(a_scales.data_ptr()),                                              \
            static_cast<float*>(b_scales.data_ptr()),                                              \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),                                   \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),                                   \
            static_cast<int*>(problem_sizes.data_ptr()),                                           \
            static_cast<int*>(problem_sizes_transpose.data_ptr()),                                 \
            transpose);                                                                            \
  }

namespace {}
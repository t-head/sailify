#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#include <cuda_runtime.h>
#include <limits>

namespace at::native {

namespace {

struct offset_t {
  int stride;
  int begin;
  __device__ int operator[](int i) const {
    return stride * (begin + i);
  }
#if CCCL_VERSION >= 3001000
  __device__ offset_t& operator+=(int i) {
    begin += i;
    return *this;
  }
#endif
};
                                          
                                                                 
                                       
                                       
                                       

                                     
                                       
                                       
                                       

                                     
                                       
                                       
                                       

                                                                   
                                                                    
                                                                 
                                                                    
                                                                
                                                                    
                                                    
                                                               
                                       

template <typename scalar_t>
__global__ void sort_postprocess_kernel(
    const scalar_t* in,
    scalar_t* out,
    int64_t* index,
    const int2* i_s_ptr,
    int nsegments,
    int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;

    int offset = segment * nsort;
    const scalar_t* in_ = in + offset;
    scalar_t* out_ = out + offset;
    int64_t* index_ = index + offset;
    const int2* i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}

}
}
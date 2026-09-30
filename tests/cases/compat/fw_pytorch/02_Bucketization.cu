#include <cuda_runtime.h>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif

namespace at::native {

                                                                                       
                                               

namespace {

template<typename input_t>
__device__ int64_t lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
                                                                                                                 
                                                                                                              
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
__device__ int64_t upper_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
                                                                                                                 
                                                                                                              
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
__global__ void searchsorted_cuda_kernel(
  output_t *data_out,
  const input_t *data_in,
  const input_t *data_bd,
  const int64_t *data_sort,
  int64_t idim_in,
  int64_t idim_bd,
  int64_t numel_in,
  bool right,
  bool is_1d_boundaries) {

  for (int64_t tid = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x; tid < numel_in; tid += blockDim.x * gridDim.x) {
                                                                              
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right ?
      lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd :
      upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd;

                                        
    data_out[tid] = pos;
  }
}

}
}
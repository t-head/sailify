#include <cuda_runtime.h>
#include <type_traits>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif

namespace at::native {

namespace {

constexpr int ALIGN_BYTES = 16;

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input),  logsum(std::log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - max_input - logsum);
}

  const AccumT max_input;
  const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

                                                                   
                                             
  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
 struct SoftMaxForwardWithMulEpilogue {
   __device__ __forceinline__ SoftMaxForwardWithMulEpilogue(AccumT max_input, AccumT sum)
     : max_input(max_input)
     , sum(sum) {}

   __device__ __forceinline__ OutT operator()(T input) const {
     return static_cast<OutT>(__expf(input - max_input) * sum);
   }

   const AccumT max_input;
   const AccumT sum;
 };




                                                                                
                                                                 
                                                                                
                                                                               
                          
                                                                
                                                                               
                                                             
                                                                                   
inline dim3 SpatialSoftMax_getGridSize(
    dim3 block, uint32_t max_active_blocks,
    uint64_t outer_size, uint64_t inner_size) {
                                                         
  uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active_blocks)
    inner_blocks = max_active_blocks;
                                                                                
  uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size)
    outer_blocks = outer_size;
  return dim3(outer_blocks, inner_blocks);
}

const int max_threads = 1024;
template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }

  __device__ __forceinline__ T combine(T a, T b) const {
    return a + b;
  }

                                                                
                           
  __device__ __forceinline__ T warp_shfl_down(T data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }

  __device__ __forceinline__ T combine(T a, T b) const {
    return a < b ? b : a;
  }

                                                                
                           
  __device__ __forceinline__ T warp_shfl_down(T data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
};

                                                      
                                                     
template<typename T, template<typename> class ReduceOp>
__forceinline__ __device__
T spatialBlockReduceX(T *shared, T val) {
  ReduceOp<T> r;
  shared += threadIdx.y * blockDim.x;

  __syncthreads();

  shared[threadIdx.x] = val;

                                           
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset)
      shared[threadIdx.x] = r(shared[threadIdx.x], shared[threadIdx.x + offset]);
    offset /= 2;
  }

  __syncthreads();

  return shared[0];
}



template <typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxBackward(
    scalar_t *gradInput, const outscalar_t *output, const outscalar_t *gradOutput,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
                                          
      if (blockDim.x > 1) {
        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += gradOutput[data_offset + d * dim_stride];
        sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      } else {
        accscalar_t sum = 0;
        for (uint32_t d = 0; d < dim_size; d++)
          sum += gradOutput[data_offset + d * dim_stride];

        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
        for (uint32_t d = 0; d < dim_size; d++) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      }
    }
  }
}


                                                                                
                                                                         
                                                                                


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template<typename T, typename AccumT>
struct SumExpfFloat
{
  __device__ __forceinline__ SumExpfFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + __expf(v - max_k);
  }

  const AccumT max_k;
};

int32_t potential_register_count(int32_t dim_size, int32_t thread_count){
                                                                                                        
  int reg_cnt = (dim_size + thread_count - 1) / thread_count;
  return reg_cnt;
}

   
                                                                                       
   
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResults(
             int classes,
             const scalar_t *input,
             outscalar_t *output,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  for (int offset = threadIdx.x; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue, typename index_t>
__device__ __forceinline__ void
WriteBpropResults(
             int classes,
             scalar_t *gradInput,
             const outscalar_t *output,
             const outscalar_t *gradOutput,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {

  index_t offset = threadIdx.x;

  index_t last = classes % (ILP * blockDim.x);

  for (; offset < classes - last; offset += blockDim.x * ILP) {
    outscalar_t tmpOutput[ILP];
    outscalar_t tmpGradOutput[ILP];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpOutput[j] = output[offset + j * blockDim.x];
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
    }

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      gradInput[offset + j * blockDim.x] = epilogue(tmpGradOutput[j], tmpOutput[j]);
    }
  }

                       
  for (; offset < classes; offset += blockDim.x) {
    gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
  }
}
}}
#include <cuda_runtime.h>
#include <type_traits>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif

constexpr float EPSILON = 1e-12;

namespace {
}namespace at::native {
                                      
           
                                      
namespace {
template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t ndim,
    int64_t n_classes,
    int64_t ignore_index) {
                                                     
  extern __shared__ unsigned char shmem[];
  accscalar_t* sh_inputs = reinterpret_cast<accscalar_t*>(shmem);
  accscalar_t* acc_weight = reinterpret_cast<accscalar_t*>(shmem + blockDim.x * sizeof(accscalar_t));

  sh_inputs[threadIdx.x] = static_cast<accscalar_t>(0);
  acc_weight[threadIdx.x] = static_cast<accscalar_t>(0);
  for (int i = threadIdx.x; i < nframe; i += blockDim.x) {
    index_t t = target[i];
    if (t != ignore_index) {
      CHECK_INDEX_IN_CLASS(t, n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
    }
  }

  __syncthreads();

  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_inputs[threadIdx.x] += sh_inputs[threadIdx.x + stride];
      acc_weight[threadIdx.x] += acc_weight[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *total_weight = static_cast<scalar_t>(acc_weight[0]);
    if (size_average) {
      *output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0]);
    } else {
      *output = static_cast<scalar_t>(sh_inputs[0]);
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_1d(
  scalar_t *grad_input,
  const scalar_t *grad_output,
  const scalar_t *weights,
  const index_t *target,
  const scalar_t *total_weight,
  bool size_average,
  int64_t n_classes,
  int64_t ignore_index
) {
  const index_t t = *target;
  if (t != ignore_index) {
    CHECK_INDEX_IN_CLASS(t, n_classes);
    const auto grad = -(size_average ? *grad_output / *total_weight : *grad_output);
    grad_input[t] = weights != nullptr ? weights[t] * grad : grad;
  }
}

template <typename T> struct bwd_index_type { using type = T; };
template<> struct bwd_index_type<uint8_t> { using type = int; };
template<> struct bwd_index_type<int64_t> { using type = uint64_t; };

}}
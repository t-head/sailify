#include <cuda_runtime.h>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif
#include <algorithm>
#include <cfloat>
#include <cmath>


namespace at::native {

namespace {

__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

  
               
                                                                     
   
 template <typename T>
__global__ void adaptivemaxgradinput(T *gradInput, const T *gradOutput, const int64_t *indices,
                             int isizeH, int isizeW,
                             int osizeH, int osizeW)
{
              
  int oh, ow;

                                             
  int o_plane = blockIdx.x;
  int i_plane = o_plane;
                               

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

                              
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

                      
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      const T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      const int64_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind);

      gradInput[argmax] += z;
    }
  }
}

  
               
                                                                     
                                                 
   
 template <typename T>
__global__ void atomicadaptivemaxgradinput(
  T *gradInput, const T *gradOutput, const int64_t *indices,
  int isizeH, int isizeW, int osizeH, int osizeW
)
{
              
  int oh, ow;

                                             
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

                              
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

                      
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      const T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      const int64_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind);

                                                                      
      gpuAtomicAddNoReturn(&(gradInput[argmax]), z);
    }
  }
}
}}
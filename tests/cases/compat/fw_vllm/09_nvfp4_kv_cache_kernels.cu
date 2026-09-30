#include <cuda_runtime.h>
#define NVFP4_ENABLE_ELTS16 1
#include <cmath>
#include <string>

namespace vllm {

                                                                                
enum class NVFP4KVScaleSearch {
  DEFAULT,
  FOUR_OVER_SIX,
};

                                                                 
                                         
                                                                 
                                                                     
  
                                                          
                                             
                                             
__device__ __forceinline__ int swizzle_scale_offset(int t, int s,
                                                    int scale_dim) {
  int s_group = scale_dim / 4;
  int swizzled_t = (t / 4) * 4 + (s / s_group);
  int swizzled_s = (s % s_group) * 4 + (t % 4);
  return swizzled_t * scale_dim + swizzled_s;
}

__device__ __forceinline__ float round_to_nearest_e2m1(float x) {
  const float ax = fabsf(x);
  float q;
                                                                          
  if (ax <= 0.25f) {
    q = 0.0f;
  } else if (ax < 0.75f) {
    q = 0.5f;
  } else if (ax <= 1.25f) {
    q = 1.0f;
  } else if (ax < 1.75f) {
    q = 1.5f;
  } else if (ax <= 2.5f) {
    q = 2.0f;
  } else if (ax < 3.5f) {
    q = 3.0f;
  } else if (ax <= 5.0f) {
    q = 4.0f;
  } else {
    q = 6.0f;
  }
  return copysignf(q, x);
}

}
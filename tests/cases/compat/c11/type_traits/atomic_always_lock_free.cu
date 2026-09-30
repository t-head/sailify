#define V(x) static_assert(x, "");
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__global__ void kernel() {
  __shared__ double tsm;
  double *dp = &tsm;
  alignas(16) [[maybe_unused]]__shared__ float tsm2[4];
  V(__atomic_always_lock_free(sizeof(double), 0))
  V(__atomic_always_lock_free(sizeof(double), dp))
  V(__atomic_always_lock_free(sizeof(__half), 0))
  V(__atomic_always_lock_free(sizeof(__half2), 0))
  V(__atomic_always_lock_free(sizeof(__nv_bfloat16), 0))
  V(__atomic_always_lock_free(sizeof(__nv_bfloat162), 0))
  V(__atomic_always_lock_free(sizeof(uint2), 0))
  V(!__atomic_always_lock_free(sizeof(dim3), 0))
}

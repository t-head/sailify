#include <cuda_runtime.h>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif

namespace at::native {

namespace detail {

                                                      
                                                                      
              
                                                                             
                                                                           
          
                                                                               
          
                                            
                                                          
                  
  
                                                                       
                                                                      
                                             
                                                                              
                                                                       
                                                                       
                                                                             
                                
static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;

static __host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline void stream_load128(uint4& val, const T* addr) {
  uint64_t low, high;
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  low = reinterpret_cast<const uint64_t*>(addr)[0];
  high = reinterpret_cast<const uint64_t*>(addr)[1];
#else
  asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
      : "=l"(low), "=l"(high)
      : "l"(addr));
#endif
  reinterpret_cast<uint64_t*>(&val)[0] = low;
  reinterpret_cast<uint64_t*>(&val)[1] = high;
}

template <typename T>
__device__ inline void stream_store128(T* addr, const uint4& val) {
  uint64_t low, high;
  low = reinterpret_cast<const uint64_t*>(&val)[0];
  high = reinterpret_cast<const uint64_t*>(&val)[1];
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  reinterpret_cast<uint64_t*>(addr)[0] = low;
  reinterpret_cast<uint64_t*>(addr)[1] = high;
#else
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

template <typename T>
static __device__ inline bool is_aligned(const void* addr) {
  return reinterpret_cast<uintptr_t>(addr) % sizeof(T) == 0;
}

template <typename T>
static __device__ inline void load128(uint4& val, const char* addr) {
  for (size_t i = 0; i < detail::BYTES_PER_THREAD / sizeof(T); ++i) {
    reinterpret_cast<T*>(&val)[i] = reinterpret_cast<const T*>(addr)[i];
  }
}

template <>
__device__ inline void load128<uint4>(uint4& val, const char* addr) {
  stream_load128(val, addr);
}

static __device__ inline void load128(uint4& val, const char* addr) {
  if (is_aligned<uint4>(addr)) {
    load128<uint4>(val, addr);
  } else if (is_aligned<int64_t>(addr)) {
    load128<uint64_t>(val, addr);
  } else if (is_aligned<uint32_t>(addr)) {
    load128<uint32_t>(val, addr);
  } else {
    load128<uint8_t>(val, addr);
  }
}

static __device__ __inline__ void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = detail::div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

static __device__ __inline__ void copy_chunk(
    char* dst,
    const char* src,
    int64_t chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  if (chunk_size < num_threads) {
    if (thread_idx < chunk_size) {
      dst[thread_idx] = src[thread_idx];
    }
    return;
  }

                                                                             
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, chunk_size, detail::BYTES_PER_THREAD, align_off, aligned_size);

  for (int64_t off = align_off + thread_idx * detail::BYTES_PER_THREAD;
       off < align_off + aligned_size;
       off += num_threads * detail::BYTES_PER_THREAD) {
    uint4 val;
                                        
    load128(val, &src[off]);
    stream_store128(&dst[off], val);
  }

                             
  if (thread_idx < align_off && thread_idx < chunk_size) {
    dst[thread_idx] = src[thread_idx];
  }
  if (align_off + aligned_size + thread_idx < chunk_size) {
    dst[align_off + aligned_size + thread_idx] =
        src[align_off + aligned_size + thread_idx];
  }
}

static __global__ void split_with_sizes_copy_out_contiguous_no_cast_kernel(
    char** dst_base_addrs,
    char** src_base_addrs,
    int64_t* split_chunk_sizes,
    int64_t* block_idx_to_split_idx,
    int64_t* blocks_cumsums,
    int64_t src_stride,
    int64_t num_chunks) {
  const int64_t split_idx = block_idx_to_split_idx[blockIdx.x];
  const int64_t split_blocks =
      blocks_cumsums[split_idx + 1] - blocks_cumsums[split_idx];
  const int64_t split_threads = split_blocks * blockDim.x;
  const int64_t split_thread_idx =
      (blockIdx.x - blocks_cumsums[split_idx]) * blockDim.x + threadIdx.x;
  const int64_t split_chunk_size = split_chunk_sizes[split_idx];

  char* dst_base_addr = dst_base_addrs[split_idx];
  char* src_base_addr = src_base_addrs[split_idx];

  for (int64_t i = blockIdx.y; i < num_chunks; i += gridDim.y) {
    copy_chunk(
        dst_base_addr + i * split_chunk_size,
        src_base_addr + i * src_stride,
        split_chunk_size,
        split_thread_idx,
        split_threads);
  }
}

}}
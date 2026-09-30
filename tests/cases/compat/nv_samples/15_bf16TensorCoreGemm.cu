

#include <assert.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <hggc_mma.h>
#include <stdio.h>


#include <helper_cuda.h>
#include <helper_functions.h>


#ifndef CPU_DEBUG

#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K


#define SHARED_MEMORY_LIMIT_64K 0
#endif


#define WARP_SIZE 32


#define M 16
#define N 16
#define K 16


#define M_TILES 512
#define N_TILES 512
#define K_TILES 512

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT awmma::mem_row_major


#define WARPS_PER_BLOCK   8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K


#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES          (CHUNK_K * K * sizeof(__nv_bfloat16))
#define WARP_COPY_BYTES           (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES     (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)


#define SKEW_BF16 16

#define checkKernelErrors(expr)                                                               \
    do {                                                                                      \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess) {                                                           \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

enum kernels {
    bf16mma_shmem_gemm_async_copy = 0, 
    bf16mma_shmem_gemm            = 1, 
    simple_bf16mma_gemm           = 2  
};

const char *kernelNames[] = {"compute_bf16gemm_async_copy", "compute_bf16gemm", "simple_wmma_bf16gemm"};

using namespace awmma;

__host__ void init_host_matrices(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = (__nv_bfloat16)(float)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = (__nv_bfloat16)(float)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = (float)(rand() % 3);
    }
}

__global__ void
compute_bf16gemm(const __nv_bfloat16 *A, const __nv_bfloat16 *B, const float *C, float *D, float alpha, float beta)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __nv_bfloat16 shmem[][CHUNK_K * K + SKEW_BF16];

    
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] + (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS
                               + (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

    
    float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    
    
    
    beta /= alpha;

    
    
    
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        
        if (block_tile_i >= M_TILES) {
            break;
        }

        
        const size_t gmem_idx                 = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        
#pragma unroll
        for (int i = 0; i < N; i++) {
            *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((int4 *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        
        
        awmma::fragment<awmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                awmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        
        
        const __nv_bfloat16 *warp_ptr =
            (warpId < (WARPS_PER_BLOCK / 2))
                ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

        
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            
            
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            
            
            const __nv_bfloat16 *lane_ptr = (warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL);

            
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                    *((int4 *)lane_ptr + (laneId % CHUNK_COPY_LINE_LANES));

                
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                awmma::fragment<awmma::matrix_a, M, N, K, __nv_bfloat16, awmma::row_major> a[WARP_COL_TILES];
                awmma::fragment<awmma::matrix_b, M, N, K, __nv_bfloat16, awmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t               shmem_idx_a = (warpId / BLOCK_ROW_WARPS) * M * BLOCK_ROW_WARPS + (i * M);
                    const __nv_bfloat16 *tile_ptr    = &shmem[shmem_idx_a][k_step * K];

                    awmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_BF16);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            
                            
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const __nv_bfloat16 *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            awmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_BF16);
                        }

                        awmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                
                
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                awmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < N; i++) {
            *((float4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((float4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
#endif
}


__global__ void simple_wmma_bf16gemm(__nv_bfloat16 *a,
                                     __nv_bfloat16 *b,
                                     float         *c,
                                     float         *d,
                                     int            m_ld,
                                     int            n_ld,
                                     int            k_ld,
                                     float          alpha,
                                     float          beta)
{
#if __CUDA_ARCH__ >= 800
    
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    
    awmma::fragment<awmma::matrix_a, M, N, K, __nv_bfloat16, awmma::row_major> a_frag;
    awmma::fragment<awmma::matrix_b, M, N, K, __nv_bfloat16, awmma::col_major> b_frag;
    awmma::fragment<awmma::accumulator, M, N, K, float>                       acc_frag;
    awmma::fragment<awmma::accumulator, M, N, K, float>                       c_frag;

    awmma::fill_fragment(acc_frag, 0.0f);

    
    for (int i = 0; i < k_ld; i += K) {
        int aCol = i;
        int aRow = warpM * M;

        int bCol = i;
        int bRow = warpN * N;

        
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            
            awmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            awmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            
            awmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    
    int cCol = warpN * N;
    int cRow = warpM * M;

    if (cRow < m_ld && cCol < n_ld) {
        awmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, awmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        
        awmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, awmma::mem_row_major);
    }
#endif
}

__host__ void matMultiplyOnHost(__nv_bfloat16 *A,
                                __nv_bfloat16 *B,
                                float         *C,
                                float          alpha,
                                float          beta,
                                int            numARows,
                                int            numAColumns,
                                int            numBRows,
                                int            numBColumns,
                                int            numCRows,
                                int            numCColumns)
{
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            float temp = 0.0;

            for (int k = 0; k < numAColumns; k++) {
                temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
            }

            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}


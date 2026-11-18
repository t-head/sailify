


#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>


#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__device__ uchar4 uint_to_uchar4(const unsigned int in)
{
    return make_uchar4(
        (in & 0x000000ff) >> 0, (in & 0x0000ff00) >> 8, (in & 0x00ff0000) >> 16, (in & 0xff000000) >> 24);
}


struct packed_result
{
    uint4 x, y, z, w;
};

__device__ packed_result get_prefix_sum(const uint4 &data, const cg::thread_block &cta)
{
    const auto tile = cg::tiled_partition<32>(cta);

    __shared__ unsigned int sums[128];
    const unsigned int      lane_id = tile.thread_rank();
    const unsigned int      warp_id = tile.meta_group_rank();

    unsigned int result[16] = {};
    {
        const uchar4 a = uint_to_uchar4(data.x);
        const uchar4 b = uint_to_uchar4(data.y);
        const uchar4 c = uint_to_uchar4(data.z);
        const uchar4 d = uint_to_uchar4(data.w);

        result[0] = a.x;
        result[1] = a.x + a.y;
        result[2] = a.x + a.y + a.z;
        result[3] = a.x + a.y + a.z + a.w;

        result[4] = b.x;
        result[5] = b.x + b.y;
        result[6] = b.x + b.y + b.z;
        result[7] = b.x + b.y + b.z + b.w;

        result[8]  = c.x;
        result[9]  = c.x + c.y;
        result[10] = c.x + c.y + c.z;
        result[11] = c.x + c.y + c.z + c.w;

        result[12] = d.x;
        result[13] = d.x + d.y;
        result[14] = d.x + d.y + d.z;
        result[15] = d.x + d.y + d.z + d.w;
    }

#pragma unroll
    for (unsigned int i = 4; i <= 7; i++)
        result[i] += result[3];

#pragma unroll
    for (unsigned int i = 8; i <= 11; i++)
        result[i] += result[7];

#pragma unroll
    for (unsigned int i = 12; i <= 15; i++)
        result[i] += result[11];

    unsigned int sum = result[15];

    
    
    
    
    
    

#pragma unroll
    for (unsigned int i = 1; i < 32; i *= 2) {
        const unsigned int n = tile.shfl_up(sum, i);

        if (lane_id >= i) {
#pragma unroll
            for (unsigned int j = 0; j < 16; j++) {
                result[j] += n;
            }

            sum += n;
        }
    }

    
    
    
    
    
    
    
    
    if (tile.thread_rank() == (tile.size() - 1)) {
        sums[warp_id] = result[15];
    }

    __syncthreads();

    if (warp_id == 0) {
        unsigned int warp_sum = sums[lane_id];

#pragma unroll
        for (unsigned int i = 1; i <= 16; i *= 2) {
            const unsigned int n = tile.shfl_up(warp_sum, i);

            if (lane_id >= i)
                warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    
    if (warp_id > 0) {
        const unsigned int blockSum = sums[warp_id - 1];

#pragma unroll
        for (unsigned int i = 0; i < 16; i++) {
            result[i] += blockSum;
        }
    }

    packed_result out;
    memcpy(&out, result, sizeof(out));
    return out;
}


__global__ void shfl_intimage_rows(const uint4 *img, uint4 *integral_image)
{
    const auto cta  = cg::this_thread_block();
    const auto tile = cg::tiled_partition<32>(cta);

    const unsigned int id = threadIdx.x;
    
    const uint4  *scanline = &img[blockIdx.x * 120];
    packed_result result   = get_prefix_sum(scanline[id], cta);

    
    
    auto idxToElem = [&result](unsigned int idx) -> const uint4 {
        switch (idx) {
        case 0:
            return result.x;
        case 1:
            return result.y;
        case 2:
            return result.z;
        case 3:
            return result.w;
        }
        return {};
    };

    
    
    
    
    
    
    
    
    
    


    const unsigned int idMask      = id & 3;
    const unsigned int idSwizzle   = (id + 2) & 3;
    const unsigned int idShift     = (id >> 2) << 4;
    const unsigned int blockOffset = blockIdx.x * 480;

    
    result.y = tile.shfl_xor(result.y, 1);
    result.z = tile.shfl_xor(result.z, 2);
    result.w = tile.shfl_xor(result.w, 3);

    
    integral_image[blockOffset + idMask + idShift] = idxToElem(idMask);
    
    integral_image[blockOffset + idSwizzle + idShift + 8] = idxToElem(idSwizzle);

    
    
    
    result.x = tile.shfl_xor(result.x, 1);
    result.y = tile.shfl_xor(result.y, 1);
    result.z = tile.shfl_xor(result.z, 1);
    result.w = tile.shfl_xor(result.w, 1);

    
    integral_image[blockOffset + idMask + idShift + 4] = idxToElem(idMask);
    
    integral_image[blockOffset + idSwizzle + idShift + 12] = idxToElem(idSwizzle);
}


__global__ void shfl_vertical_shfl(unsigned int *img, int width, int height)
{
    __shared__ unsigned int sums[32][9];
    int                     tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int lane_id = tidx % 8;
    
    
    unsigned int stepSum = 0;
    unsigned int mask    = 0xffffffff;

    sums[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    for (int step = 0; step < 135; step++) {
        unsigned int  sum = 0;
        unsigned int *p   = img + (threadIdx.y + step * 8) * width + tidx;

        sum                            = *p;
        sums[threadIdx.x][threadIdx.y] = sum;
        __syncthreads();

        
        
        
        
        int partial_sum = 0;
        int j           = threadIdx.x % 8;
        int k           = threadIdx.x / 8 + threadIdx.y * 4;

        partial_sum = sums[k][j];

        for (int i = 1; i <= 8; i *= 2) {
            int n = __shfl_up_sync(mask, partial_sum, i, 32);

            if (lane_id >= i)
                partial_sum += n;
        }

        sums[k][j] = partial_sum;
        __syncthreads();

        if (threadIdx.y > 0) {
            sum += sums[threadIdx.x][threadIdx.y - 1];
        }

        sum += stepSum;
        stepSum += sums[threadIdx.x][blockDim.y - 1];
        __syncthreads();
        *p = sum;
    }
}


__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
    extern __shared__ int sums[];
    int                   id      = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int                   lane_id = id % warpSize;
    
    int warp_id = threadIdx.x / warpSize;

    
    
    
    int value = data[id];

    
    
    
    
    
    

#pragma unroll
    for (int i = 1; i <= width; i *= 2) {
        unsigned int mask = 0xffffffff;
        int          n    = __shfl_up_sync(mask, value, i, width);

        if (lane_id >= i)
            value += n;
    }

    
    

    
    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }

    __syncthreads();

    
    
    
    
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        int warp_sum = sums[lane_id];

        int mask = (1 << (blockDim.x / warpSize)) - 1;
        for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
            int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

            if (lane_id >= i)
                warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    
    
    int blockSum = 0;

    if (warp_id > 0) {
        blockSum = sums[warp_id - 1];
    }

    value += blockSum;

    
    data[id] = value;

    
    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = value;
    }
}


__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int            id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len)
        return;

    if (threadIdx.x == 0) {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


unsigned int verifyDataRowSums(unsigned int *h_image, int w, int h)
{
    unsigned int diff = 0;

    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            int gold = i + 1;
            diff += abs(static_cast<int>(gold) - static_cast<int>(h_image[j * w + i]));
        }
    }

    return diff;
}


bool shuffle_integral_image_test()
{
    char         *d_data;
    unsigned int *h_image;
    unsigned int *d_integral_image;
    int           w          = 1920;
    int           h          = 1080;
    int           n_elements = w * h;
    int           sz         = sizeof(unsigned int) * n_elements;

    printf("\nComputing Integral Image Test on size %d x %d synthetic data\n", w, h);
    printf("---------------------------------------------------\n");
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_image), sz));
    
    memset(h_image, 0, sz);

    
    int blockSize = iDivUp(w, 16);
    
    int gridSize = h;

    
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_data), sz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_integral_image), n_elements * sizeof(int) * 4));
    checkCudaErrors(cudaMemset(d_data, 1, sz));
    checkCudaErrors(cudaMemset(d_integral_image, 0, sz));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float        et = 0;
    unsigned int err;

    
    cudaEventRecord(start);
    shfl_intimage_rows<<<gridSize, blockSize>>>(reinterpret_cast<uint4 *>(d_data),
                                                reinterpret_cast<uint4 *>(d_integral_image));
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&et, start, stop));
    printf("Method: Fast  Time (GPU Timer): %f ms ", et);

    
    checkCudaErrors(cudaMemcpy(h_image, d_integral_image, sz, cudaMemcpyDeviceToHost));
    err = verifyDataRowSums(h_image, w, h);
    printf("Diff = %d\n", err);

    
    dim3 blockSz(32, 8);
    dim3 testGrid(w / blockSz.x, 1);

    cudaEventRecord(start);
    shfl_vertical_shfl<<<testGrid, blockSz>>>((unsigned int *)d_integral_image, w, h);
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&et, start, stop));
    printf("Method: Vertical Scan  Time (GPU Timer): %f ms ", et);

    
    checkCudaErrors(cudaMemcpy(h_image, d_integral_image, sz, cudaMemcpyDeviceToHost));
    printf("\n");

    int finalSum = h_image[w * h - 1];
    printf("CheckSum: %d, (expect %dx%d=%d)\n", finalSum, w, h, w * h);

    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_integral_image));
    checkCudaErrors(cudaFreeHost(h_image));
    
    
    return (finalSum == w * h) ? true : false;
}

#include <cuda_runtime.h>


#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>   
#include <helper_image.h>  
#include <helper_string.h> 

const char *sSDKsample = "Transpose";


#define TILE_DIM   32
#define BLOCK_ROWS 16


int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR    = TILE_DIM;

#define FLOOR(a, b) (a - (a % b))


int MAX_TILES = (FLOOR(MATRIX_SIZE_X, 512) * FLOOR(MATRIX_SIZE_Y, 512)) / (TILE_DIM * TILE_DIM);


#define NUM_REPS 100


__global__ void copy(float *odata, float *idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = xIndex + width * yIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index + i * width] = idata[index + i * width];
    }
}

__global__ void copySharedMem(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = xIndex + width * yIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && yIndex < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
            tile[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
        }
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < height && yIndex < width) {
            odata[index + i * width] = tile[threadIdx.y + i][threadIdx.x];
            odata[index + i * width] = tile[threadIdx.y + i][threadIdx.x];
        }
    }
}


__global__ void transposeNaive(float *odata, float *idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i] = idata[index_in + i * width];
    }
}


__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}


__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}


__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else {
        int bid    = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    
    

    int xIndex   = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}


__global__ void transposeFineGrained(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float block[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index  = xIndex + (yIndex)*width;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        block[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index + i * height] = block[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeCoarseGrained(float *odata, float *idata, int width, int height)
{
    
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float block[TILE_DIM][TILE_DIM + 1];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        block[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = block[threadIdx.y + i][threadIdx.x];
    }
}


void computeTransposeGold(float *gold, float *idata, const int size_x, const int size_y)
{
    for (int y = 0; y < size_y; ++y) {
        for (int x = 0; x < size_x; ++x) {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}

void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &size_x, int &size_y, int max_tile_dim)
{
    
    
    if (checkCmdLineFlag(argc, (const char **)argv, "dimX")) {
        size_x = getCmdLineArgumentInt(argc, (const char **)argv, "dimX");

        if (size_x > max_tile_dim) {
            printf("> MatrixSize X = %d is greater than the recommended size = %d\n", size_x, max_tile_dim);
        }
        else {
            printf("> MatrixSize X = %d\n", size_x);
        }
    }
    else {
        size_x = max_tile_dim;
        size_x = FLOOR(size_x, 512);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dimY")) {
        size_y = getCmdLineArgumentInt(argc, (const char **)argv, "dimY");

        if (size_y > max_tile_dim) {
            printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", size_y, max_tile_dim);
        }
        else {
            printf("> MatrixSize Y = %d\n", size_y);
        }
    }
    else {
        size_y = max_tile_dim;
        size_y = FLOOR(size_y, 512);
    }
}

void showHelp()
{
    printf("\n%s : Command line options\n", sSDKsample);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
    printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

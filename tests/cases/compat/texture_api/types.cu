// CUDA texture types in CUDA spelling; sailify rewrites the include to the
// PPU_SDK hggc_texture_types.h spelling (the C++ texture<T> class lives in
// hggc_texture_types.hpp instead).  Needs PPU SDK >= 2.3 to compile.
#include <texture_types.h>

TEST(texture_api, c_types) {
    struct cudaTextureDesc desc;
    desc.addressMode[0] = cudaAddressModeWrap;
    desc.addressMode[1] = cudaAddressModeClamp;
    desc.filterMode = cudaFilterModePoint;
    desc.readMode = cudaReadModeElementType;
    desc.normalizedCoords = 0;
    cudaTextureObject_t obj = 0;
    (void)desc;
    (void)obj;
}

TEST(texture_api, texture_reference) {
    struct textureReference ref;
    ref.normalized = 0;
    ref.filterMode = cudaFilterModeLinear;
    (void)ref;
}

// CUDA sanitizer CBID enum values in CUDA spelling (cu*/cuda* suffixes);
// converted to the 2v3 hggc names (hg*/hggc*) by sailify, so per-entry
// mappings are exercised.  Needs PPU SDK >= 2.3 to compile.
#include <sanitizer.h>

TEST(sanitizer_api, cbid_driver_values) {
    SANITIZER_CBID_DRIVER_API id = SANITIZER_CBID_DRIVER_API_INVALID;
    unsigned v[] = {
        (unsigned)SANITIZER_CBID_DRIVER_API_cuInit,
        (unsigned)SANITIZER_CBID_DRIVER_API_cuLaunchKernel,
        (unsigned)SANITIZER_CBID_DRIVER_API_cuLaunchKernel_ptsz,
        (unsigned)SANITIZER_CBID_DRIVER_API_cuCompilePtx,
        (unsigned)SANITIZER_CBID_DRIVER_API_SIZE,
    };
    (void)id;
    (void)v;
}

TEST(sanitizer_api, cbid_runtime_values) {
    SANITIZER_CBID_RUNTIME_API id = SANITIZER_CBID_RUNTIME_API_INVALID;
    unsigned v[] = {
        (unsigned)SANITIZER_CBID_RUNTIME_API_cudaDriverGetVersion,
        (unsigned)SANITIZER_CBID_RUNTIME_API_cudaMemcpyAsync,
        (unsigned)SANITIZER_CBID_RUNTIME_API_cudaMemcpyAsync_ptsz,
        (unsigned)SANITIZER_CBID_RUNTIME_API_cudaSignalExternalSemaphoresAsync,
        (unsigned)SANITIZER_CBID_RUNTIME_API_SIZE,
    };
    (void)id;
    (void)v;
}

TEST(sanitizer_api, cbid_resource_values) {
    unsigned v[] = {
        (unsigned)SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_FINISHED,
        (unsigned)SANITIZER_CBID_RESOURCE_MODULE_LOADED,
    };
    (void)v;
}

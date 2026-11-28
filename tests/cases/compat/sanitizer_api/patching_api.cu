// CUDA compute-sanitizer patching enums in CUDA spelling (several differ
// from the PPU_SDK 2v3 spellings by more than the HGGC_ infix).  Converted
// by sailify; needs PPU SDK >= 2.3 to compile.
#include <sanitizer.h>

TEST(sanitizer_api, patching_enums) {
    Sanitizer_InstructionId inst = SANITIZER_INSTRUCTION_DEVICE_ALIGNED_MALLOC;
    Sanitizer_DeviceMemoryFlags mem =
        (Sanitizer_DeviceMemoryFlags)SANITIZER_MEMORY_DEVICE_FLAG_READ;
    Sanitizer_CacheControlInstructionKind cc =
        SANITIZER_CACHE_CONTROL_L1_PREFETCH;
    Sanitizer_WarpgroupMMAAsyncFlags wf =
        (Sanitizer_WarpgroupMMAAsyncFlags)SANITIZER_WARPGROUP_MMA_ASYNC_FLAG_COMMIT_GROUP;
    Sanitizer_FunctionLoadedStatus status = SANITIZER_FUNCTION_NOT_LOADED;
    Sanitizer_CallbackDomain domain = SANITIZER_CB_DOMAIN_LAUNCH;
    Sanitizer_MemcpyDirection dir = SANITIZER_MEMCPY_DIRECTION_HOST_TO_DEVICE;
    (void)inst;
    (void)mem;
    (void)cc;
    (void)wf;
    (void)status;
    (void)domain;
    (void)dir;
}

TEST(sanitizer_api, patching_callback_types) {
    SanitizerCallbackBulkCopyGlobalToShared cb1 = nullptr;
    SanitizerCallbackBulkCopySharedToGlobal cb2 = nullptr;
    (void)cb1;
    (void)cb2;
}

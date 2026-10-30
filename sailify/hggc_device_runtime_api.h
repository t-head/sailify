#ifndef HGGC_DEVICE_RUNTIME_API_H
#define HGGC_DEVICE_RUNTIME_API_H

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

/* The CUDA-named runtime headers are spelled differently across SDK
 * distributions: the official SDK keeps the unbranded names (host_defines.h,
 * driver_types.h, vector_types.h) in its include root, while the mix-sdk
 * distribution brands them (hggc_driver_types.h, ...) and only ships
 * host_defines under hgrt/.  Probe for each spelling so the compat layer
 * works against either tree.  hgrt/host_defines.h is preferred over
 * hggc_host_defines.h because it defines __HGGCRT_HOST_DEFINES_H__, which
 * thrust's guarded_hggc_runtime_api.h checks before #undef-ing __host__ /
 * __device__. */
#if __has_include(<hgrt/host_defines.h>)
#include <hgrt/host_defines.h>
#elif __has_include(<hggc_host_defines.h>)
#include "hggc_host_defines.h"
#elif __has_include(<host_defines.h>)
#include "host_defines.h"
#endif

#if __has_include(<hggc_driver_types.h>)
#include "hggc_driver_types.h"
#elif __has_include(<driver_types.h>)
#include "driver_types.h"
#endif

#include "hggc.h"

#if __has_include(<hggc_vector_types.h>)
#include "hggc_vector_types.h"
#elif __has_include(<vector_types.h>)
#include "vector_types.h"
#endif

static __inline__ __device__ hggcGraphExec_t hggcGetCurrentGraphExec(void)
{
    printf("hggcGetCurrentGraphExec is not supported.\n");
    __builtin_trap();
    return 0;
}

static __inline__ __device__ hggcError_t hggcGraphKernelNodeSetEnabled(hggcGraphDeviceNode_t node, bool enable)
{
    (void)node; (void)enable;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ hggcError_t hggcGraphKernelNodeSetGridDim(hggcGraphDeviceNode_t node, dim3 gridDim)
{
    (void)node; (void)gridDim;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ hggcError_t hggcGraphKernelNodeUpdatesApply(const hggcGraphKernelNodeUpdate *updates, size_t updateCount)
{
    (void)updates; (void)updateCount;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ void hggcTriggerProgrammaticLaunchCompletion(void)
{
    printf("hggcTriggerProgrammaticLaunchCompletion is not supported.\n");
    __builtin_trap();
}

static __inline__ __device__ void hggcGridDependencySynchronize(void)
{
    printf("hggcGridDependencySynchronize is not supported.\n");
    __builtin_trap();
}

/* hggcCGGetIntrinsicHandle, hggcCGSynchronize, hggcCGGetSize, hggcCGGetRank: real definitions in hgrt/__cg_helper.h (force-included via hggc_runtime.h); stubs here would redefine them. */
static __inline__ __device__ hggcError_t hggcCGSynchronizeGrid(unsigned long long handle, unsigned int flags)
{
    (void)handle; (void)flags;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ hggcError_t hggcGraphKernelNodeSetParam(hggcGraphDeviceNode_t node, size_t offset, const void *value, size_t size)
{
    (void)node; (void)offset; (void)value; (void)size;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ void *hggcGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize)
{
    (void)func; (void)gridDimension; (void)blockDimension; (void)sharedMemSize;
    printf("hggcGetParameterBufferV2 is not supported.\n");
    __builtin_trap();
    return 0;
}

static __inline__ __device__ hggcError_t hggcLaunchDeviceV2(void *parameterBuffer, hggcStream_t stream)
{
    (void)parameterBuffer; (void)stream;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ hggcError_t hggcLaunchDeviceV2_ptsz(void *parameterBuffer, hggcStream_t stream)
{
    (void)parameterBuffer; (void)stream;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}

static __inline__ __device__ hggcError_t hggcLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, hggcStream_t stream)
{
    (void)func; (void)parameterBuffer; (void)gridDimension; (void)blockDimension; (void)sharedMemSize; (void)stream;
    return (hggcError_t)HGGC_ERROR_NOT_SUPPORTED;
}
#endif

#ifndef HGGC_DEVICE_RUNTIME_API_H
#define HGGC_DEVICE_RUNTIME_API_H

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#include "host_defines.h"
#include "driver_types.h"
#include "hggc.h"
#include "vector_types.h"

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

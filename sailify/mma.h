/* Forward to a later mma.h (e.g. one provided by the project) when one
   exists after this shim on the include path; otherwise fall back to
   hggc_mma.h. Only the fallback is guarded: chained shims must all keep
   forwarding, and the last one in the chain performs the guarded
   fallback. */
#ifdef __has_include_next
#if __has_include_next(<mma.h>)
#include_next <mma.h>
#else
#ifndef MMA_H_SHIM
#define MMA_H_SHIM
#include <hggc_mma.h>
#endif
#endif
#else
#ifndef MMA_H_SHIM
#define MMA_H_SHIM
#include <hggc_mma.h>
#endif
#endif

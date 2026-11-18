#ifndef P_HDR03_H
#define P_HDR03_H
#include <stddef.h>
static inline void probeNeverCalled(void) { void* p = 0; (void)cudaMalloc(&p, 4); }
#endif

#include "kernel_host.h"
typedef char kernel_ok[(chainLaunchHelper() == 7) ? 1 : -1];
void kernelAnchor(void) {
    int h = 0;
    chainKernel<<<1, 1>>>(&h);
}

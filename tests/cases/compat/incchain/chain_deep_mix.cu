#include "deep_chain.h"
#include "kernel_host.h"
typedef char mix_ok[(sizeof(DeepNode) > 0) ? 1 : -1];
int mixAnchor(void) {
    return chainLaunchHelper() + (int)sizeof(DeepNode);
}

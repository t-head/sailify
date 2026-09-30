#define INCCHAIN_GATE
#include "gate_a.h"
typedef char gate_ok[(sizeof(Gated) > 0) ? 1 : -1];
int gateAnchor(void) {
    Gated g;
    g.gated_v = 5;
    return g.gated_v;
}

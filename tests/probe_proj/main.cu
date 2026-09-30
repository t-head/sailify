#include <p_hdr02.h>
#include <p_hdr03.h>
int probeMain(void) { P02 v; v.b = P_VAL01; return v.b + sizeof(P01); }

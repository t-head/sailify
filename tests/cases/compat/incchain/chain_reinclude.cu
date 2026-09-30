#include "chain_h01.h"
#include "chain_h01.h"
#include "chain_h02.h"
typedef char reinclude_ok[(sizeof(C01) > 0) ? 1 : -1];
int reincludeAnchor(void) {
    return (int)sizeof(C01);
}

#include "chain_h01.h"
typedef char ok01[(sizeof(C01) > 0) ? 1 : -1];
typedef char ok02[(sizeof(C02) > 0) ? 1 : -1];
typedef char ok03[(sizeof(C03) > 0) ? 1 : -1];
typedef char ok04[(sizeof(C04) > 0) ? 1 : -1];
typedef char ok05[(sizeof(C05) > 0) ? 1 : -1];
typedef char ok06[(sizeof(C06) > 0) ? 1 : -1];
typedef char ok07[(sizeof(C07) > 0) ? 1 : -1];
typedef char ok08[(sizeof(C08) > 0) ? 1 : -1];
typedef char ok09[(sizeof(C09) > 0) ? 1 : -1];
typedef char ok10[(sizeof(C10) > 0) ? 1 : -1];
int chainDepthAnchor(const C01& c) {
    return c.v01 + c.next.v02 + c.next.next.v03;
}

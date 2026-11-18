#ifndef INCCHAIN_DEEP_CHAIN_H
#define INCCHAIN_DEEP_CHAIN_H
#include "chain_h01.h"
#include "diamond_left.h"
struct DeepNode {
    C01 chain;
    DLeft left;
    int depth;
};
#endif

#include "deep_chain.h"
typedef char nested_ok[(sizeof(DeepNode) > 0) ? 1 : -1];
int nestedAnchor(DeepNode& n) {
    return n.chain.v01 + n.depth + n.left.left_v;
}

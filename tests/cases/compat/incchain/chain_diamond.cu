#include "diamond_left.h"
#include "diamond_right.h"
struct DBoth : DLeft, DRight {
    int both_v;
};
typedef char diamond_ok[(sizeof(DBoth) > 0) ? 1 : -1];
int diamondAnchor(DBoth& d) {
    d.both_v = 3;
    d.left_v = 1;
    d.right_v = 2;
    return d.both_v;
}

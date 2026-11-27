// CUDA CCCL public macros in CUDA spelling; sailify rewrites them to the
// PPU_SDK 2v3 SAIL_ spellings.  Needs PPU SDK >= 2.3 to compile.
#include <thrust/version.h>
#include <cub/version.cuh>
#include <cub/util_namespace.cuh>
#include <cub/thread/thread_operators.cuh>

CUB_NAMESPACE_BEGIN
struct cccl_macro_pod {
    int v;
};
CUB_NAMESPACE_END

TEST(cccl_macro_api, version_macros) {
    cub::cccl_macro_pod p;
    p.v = THRUST_VERSION + CUB_VERSION;
}

TEST(cccl_macro_api, ns_qualifier) {
    CUB_NS_QUALIFIER::Min min_op;
    cub::cccl_macro_pod p;
    p.v = min_op(1, 2);
}

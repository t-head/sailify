import argparse
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from map_flags import translate


def _report(npass, nfail):
    print("API_PATTERNS_SUMMARY pass=" + str(npass) + " fail=" + str(nfail))
    return 0 if nfail == 0 else 1


def test_pytorch_codegen_string_conversion():
    from sailify.sailify_python import _PPU_MAP, _PPU_TRIE

    rx = re.compile(rf"({_PPU_TRIE.export_to_regex()})(?=\W)")

    def sailify_str(s):
        return rx.sub(lambda m: _PPU_MAP[m.group(0)], s)

    cases = [
        ("static CUfunction kernel_0 = nullptr;", ["HGfunction"]),
        ("cudaStream_t stream_,", ["hggcStream_t"]),
        ("CUresult code = cuGetErrorString(code, &msg);", ["HGresult", "hgGetErrorString"]),
        ("if (code != CUDA_SUCCESS)", ["HGGC_SUCCESS"]),
        ("CUtensorMap m;", ["HGtensorMap"]),
        ("CUdeviceptr ptr;", ["HGdeviceptr"]),
        ("#if CUDA_VERSION >= 12000", ["COMPATIBLE_VERSION"]),
        ("#ifdef __CUDA_ARCH__\nint x;\n#endif", ["COMPATIBLE_ARCH"]),
        ("cudaDriverGetVersion(&v);", ["compatibleDriverGetVersion"]),
        ("cudaMalloc(&p, n);", ["hggcMalloc"]),
        ("cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, stream);",
         ["hggcMemcpyAsync", "hggcMemcpyHostToDevice"]),
        ("CUresult rc = cuLaunchKernel(f, g, b, 0, 0, stream, args);",
         ["HGresult", "hgLaunchKernel"]),
    ]
    failures = []
    for src, expected in cases:
        got = sailify_str(src)
        for e in expected:
            if e not in got:
                failures.append(src + " missing " + e)
        if sailify_str(got) != got:
            failures.append(src + " not idempotent")
    untouched = [
        "at::cuda::getCurrentCUDAStream();",
        "auto s = getStreamFromExternalMasqueradingAsCUDA();",
        "std::string msg = \"CUDA driver error: \";",
    ]
    for src in untouched:
        if sailify_str(src) != src:
            failures.append(src + " should stay untouched")
    return failures


def test_write_compat_wrapper_header(tmp):
    from sailify.sailify_python import write_compat_wrapper_header

    compat_dir = os.path.join(tmp, "torch_ppu_compat")
    write_compat_wrapper_header(compat_dir)
    failures = []
    header = os.path.join(compat_dir, "compatible_wrapper.h")
    if not os.path.isfile(header):
        failures.append("compatible_wrapper.h not written to " + compat_dir)
        return failures
    with open(header) as fh:
        content = fh.read()
    for needle in ("COMPATIBLE_VERSION", "COMPATIBLE_ARCH", "compatibleDriverGetVersion"):
        if needle not in content:
            failures.append("compatible_wrapper.h missing " + needle)
    return failures


_KERNEL_CU = """#include <cuda_runtime.h>
#include "../include/util.cuh"

__global__ void k(float* out) {
    float* p;
    cudaMalloc(&p, 16);
    util_scale(p, out[threadIdx.x]);
    cudaFree(p);
}
"""

_UTIL_CUH = """#pragma once
__device__ __forceinline__ void util_scale(float* p, float v) {
    *p = v * 2.0f;
}
"""

_UTIL2_CUH = """#pragma once
__device__ __forceinline__ void util_step(float* p, float v) {
    cudaMallocManaged(&p, 16);
}
"""


def test_sglang_jit_mirror(tmp):
    from sailify.sailify_python import sailify_extra_files_recursive

    mirror = os.path.join(tmp, "mirror")
    os.makedirs(os.path.join(mirror, "csrc"), exist_ok=True)
    os.makedirs(os.path.join(mirror, "include"), exist_ok=True)
    kernel = os.path.join(mirror, "csrc", "kernel.cu")
    util = os.path.join(mirror, "include", "util.cuh")
    with open(kernel, "w") as fh:
        fh.write(_KERNEL_CU)
    with open(util, "w") as fh:
        fh.write(_UTIL_CUH)

    results = sailify_extra_files_recursive(
        output_directory=mirror,
        extra_files=[kernel],
        header_include_dirs=[],
    )
    failures = []
    kernel_abs = os.path.abspath(kernel)
    util_abs = os.path.abspath(util)
    if kernel_abs not in results:
        failures.append("result dict not keyed by absolute input path")
        return failures
    res = results[kernel_abs]
    if res.status != "ok":
        failures.append("kernel status " + str(res.status))
    if os.path.abspath(res.ppuified_path) != kernel_abs:
        failures.append("in-place conversion expected, got " + res.ppuified_path)
    if util_abs not in results:
        failures.append("include closure header not discovered")
    with open(kernel) as fh:
        text = fh.read()
    if "cudaMalloc" in text or "hggcMalloc" not in text:
        failures.append("kernel.cu not converted in place")
    with open(util) as fh:
        text = fh.read()
    if "cudaMalloc" in text:
        failures.append("util.cuh not converted in place")
    compat = os.path.join(mirror, ".ppu_compat", "compatible_wrapper.h")
    if not os.path.isfile(compat):
        failures.append(".ppu_compat not generated in output_directory")
    return failures


def test_pytorch_cpp_extension_jit(tmp):
    from sailify.sailify_python import sailify_extra_files_recursive

    build_dir = os.path.join(tmp, "torch_ext_build")
    src_dir = os.path.join(tmp, "torch_ext_src")
    hdr_dir = os.path.join(tmp, "torch_ext_hdrs")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(hdr_dir, exist_ok=True)
    src = os.path.join(src_dir, "ext_kernel.cu")
    hdr = os.path.join(hdr_dir, "ext_util.cuh")
    with open(src, "w") as fh:
        fh.write(_KERNEL_CU.replace('"../include/util.cuh"', "<ext_util.cuh>"))
    with open(hdr, "w") as fh:
        fh.write(_UTIL2_CUH)

    results = sailify_extra_files_recursive(
        output_directory=build_dir,
        extra_files=[src],
        header_include_dirs=[hdr_dir],
        show_detailed=True,
    )
    failures = []
    src_abs = os.path.abspath(src)
    hdr_abs = os.path.abspath(hdr)
    if src_abs not in results:
        failures.append("result dict not keyed by absolute input path")
        return failures
    res = results[src_abs]
    if res.status != "ok":
        failures.append("source status " + str(res.status))
    if os.path.splitext(os.path.abspath(res.ppuified_path))[1] != ".cu":
        failures.append("extension not preserved: " + res.ppuified_path)
    if hdr_abs not in results:
        failures.append("-I include dir header not discovered")
    with open(src) as fh:
        if "cudaMalloc" in fh.read():
            failures.append("ext_kernel.cu not converted")
    with open(hdr) as fh:
        text = fh.read()
    if "cudaMallocManaged" in text or "hggcMallocManaged" not in text:
        failures.append("ext_util.cuh not converted")
    compat = os.path.join(build_dir, ".ppu_compat", "compatible_wrapper.h")
    if not os.path.isfile(compat):
        failures.append(".ppu_compat not generated in build_directory")
    return failures


def test_sglang_aot_preprocessor_direct(tmp):
    from sailify.sailify_python import (
        _discover_includes,
        _ensure_ppu_compat,
        preprocessor,
    )

    upstream = os.path.join(tmp, "upstream")
    staging = os.path.join(tmp, "staging")
    os.makedirs(os.path.join(upstream, "csrc"), exist_ok=True)
    src = os.path.join(upstream, "csrc", "aot_kernel.cu")
    hdr = os.path.join(upstream, "csrc", "aot_util.cuh")
    with open(src, "w") as fh:
        fh.write(_KERNEL_CU.replace('"../include/util.cuh"', '"aot_util.cuh"'))
    with open(hdr, "w") as fh:
        fh.write(_UTIL_CUH)

    failures = []
    stats = {}
    res = preprocessor(
        output_directory=staging,
        filepath=src,
        all_files=[],
        stats=stats,
        clean_ctx=None,
        show_progress=False,
        project_directory=upstream,
    )
    if res.status != "ok":
        failures.append("preprocessor status " + str(res.status))
    staged = os.path.join(staging, "csrc", "aot_kernel.cu")
    if not os.path.isfile(staged):
        failures.append("staged output missing at " + staged)
    else:
        with open(staged) as fh:
            text = fh.read()
        if "cudaMalloc" in text or "hggcMalloc" not in text:
            failures.append("staged kernel not converted")
    if stats.get("ppuified") != [src]:
        failures.append("stats['ppuified'] != [source path]")

    visited = {src}
    discovered = []
    _discover_includes(src, [os.path.join(upstream, "csrc")], visited, discovered)
    if hdr not in discovered:
        failures.append("_discover_includes missed aot_util.cuh")

    hdr_stats = {}
    hdr_res = preprocessor(
        output_directory=staging,
        filepath=hdr,
        all_files=[],
        stats=hdr_stats,
        clean_ctx=None,
        show_progress=False,
        project_directory=upstream,
    )
    if hdr_res.status != "ok":
        failures.append("header preprocessor status " + str(hdr_res.status))
    staged_hdr = os.path.join(staging, "csrc", "aot_util.cuh")
    if not os.path.isfile(staged_hdr):
        failures.append("staged header missing at " + staged_hdr)

    _ensure_ppu_compat(staging)
    if not os.path.isfile(os.path.join(staging, ".ppu_compat", "compatible_wrapper.h")):
        failures.append("_ensure_ppu_compat did not generate .ppu_compat")
    return failures


def test_sdk22_mapping_state_toggles_with_version():
    import sailify.sailify_python as sp

    # hermetic SDK: declares every 2.2-era target spelling
    sdk_inc = os.path.join(tempfile.mkdtemp(prefix="sailify_sdk22_state_"), "include")
    os.makedirs(sdk_inc, exist_ok=True)
    with open(os.path.join(sdk_inc, "hggc_all.h"), "w") as fh:
        for _src, dst in sp.PPU_SDK_2_2_IDENTIFIER_MAP.items():
            fh.write("int %s;\n" % dst)

    failures = []
    orig_dirs = sp.ppu_sdk_include_dirs
    try:
        sp.ppu_sdk_include_dirs = lambda: [sdk_inc]
        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        if not sp._SDK22_MAP or sp._SDK22_RE is None:
            failures.append("SDK 2.2 mappings not enabled for version (2, 2, 0)")
        elif sp._SDK22_RE.search("__NV_THREAD_SCOPE_THREAD") is None:
            failures.append("SDK 2.2 regex does not match a mapped identifier")
        elif len(sp._SDK22_MAP) != len(sp.PPU_SDK_2_2_IDENTIFIER_MAP):
            failures.append("probe dropped pairs whose targets the fake SDK declares")

        sp._maybe_enable_sdk_2_2_mappings((2, 1, 5))
        if sp._SDK22_MAP or sp._SDK22_RE is not None:
            failures.append("SDK 2.2 mappings not reset for version (2, 1, 5)")

        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        if not sp._SDK22_MAP or sp._SDK22_RE is None:
            failures.append("SDK 2.2 mappings not re-enabled after reset")

        sp._maybe_enable_sdk_2_2_mappings(None)
        if sp._SDK22_MAP or sp._SDK22_RE is not None:
            failures.append("SDK 2.2 mappings not reset for unknown version")
    finally:
        sp.ppu_sdk_include_dirs = orig_dirs
        sp._maybe_enable_sdk_2_2_mappings(None)
    return failures


def test_sdk22_bf16_pipeline_spelling_rewrite(tmp):
    import sailify.sailify_python as sp

    src_dir = os.path.join(tmp, "sdk22_spelling")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "k.cu")

    # wt_v3-style SDK: the declared spellings are __hg_ for the bf16
    # conversions and __hggc_pipeline_ for the pipeline primitives
    wt_inc = os.path.join(tmp, "sdk_wt", "include")
    os.makedirs(wt_inc, exist_ok=True)
    with open(os.path.join(wt_inc, "hggc_bf16.hpp"), "w") as fh:
        for name in ("__hg_bfloat162float", "__hg_float2bfloat16_rn",
                     "__hggc_pipeline_commit", "__hggc_pipeline_wait_prior"):
            fh.write("int %s;\n" % name)
    # flat-style SDK: only the unprefixed CUDA spellings exist
    flat_inc = os.path.join(tmp, "sdk_flat", "include")
    os.makedirs(flat_inc, exist_ok=True)
    with open(os.path.join(flat_inc, "hggc_bf16.hpp"), "w") as fh:
        for name in ("__bfloat162float", "__float2bfloat16_rn",
                     "__pipeline_commit", "__pipeline_wait_prior"):
            fh.write("int %s;\n" % name)

    def write_case():
        with open(src, "w") as fh:
            fh.write(
                "float f(__ppu_bfloat16 a) { return __bfloat162float(a); }\n"
                "int p; __pipeline_commit(); __pipeline_wait_prior(0);\n"
                "auto q = __float2bfloat16_rn(1.0f);\n"
                "int untouched = my__bfloat162float_x;\n"
                "float nat = __low2float(a);\n"
            )

    failures = []
    orig_dirs = sp.ppu_sdk_include_dirs
    try:
        # wt_v3 layout: CUDA spellings rewrite to the declared targets
        # (__hg_ bf16 conversions, __hggc_pipeline_ primitives)
        sp.ppu_sdk_include_dirs = lambda: [wt_inc]
        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        for cuda_name, ppu_name in (
            ("__bfloat162float", "__hg_bfloat162float"),
            ("__pipeline_commit", "__hggc_pipeline_commit"),
            ("__pipeline_wait_prior", "__hggc_pipeline_wait_prior"),
            ("__float2bfloat16_rn", "__hg_float2bfloat16_rn"),
        ):
            if ppu_name not in text:
                failures.append("SDK 2.2 spelling rewrite missing: " + cuda_name)
            if re.search(r"(?<!\w)" + re.escape(cuda_name) + r"(?!\w)", text):
                failures.append("CUDA spelling survived: " + cuda_name)
        if "my__bfloat162float_x" not in text:
            failures.append("rewrite touched a partial identifier")
        if "__low2float" not in text:
            failures.append("rewrite altered a name the SDK provides unprefixed")

        # flat layout: the __hggc_ targets are not declared, so the CUDA
        # spellings must survive (rewriting would be an undeclared identifier)
        sp.ppu_sdk_include_dirs = lambda: [flat_inc]
        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        for cuda_name in ("__bfloat162float", "__pipeline_commit",
                          "__pipeline_wait_prior", "__float2bfloat16_rn"):
            if not re.search(r"(?<!\w)" + re.escape(cuda_name) + r"(?!\w)", text):
                failures.append("flat-layout SDK lost the CUDA spelling: " + cuda_name)
        if "__hggc_" in text:
            failures.append("flat-layout SDK produced __hggc_ spellings")
    finally:
        sp.ppu_sdk_include_dirs = orig_dirs
        sp._maybe_enable_sdk_2_2_mappings(None)
    return failures


def test_prefixed_core_include_probe(tmp):
    import sailify.sailify_python as sp

    sdk_inc = os.path.join(tmp, "sdk_include")
    os.makedirs(sdk_inc, exist_ok=True)
    src_dir = os.path.join(tmp, "prefixed_inc")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "k.cu")

    def write_case():
        with open(src, "w") as fh:
            fh.write(
                "#include <vector_types.h>\n"
                "#include <driver_types.h>\n"
                "#include <some_dir/vector_functions.h>\n"
                "#include <vector_types.h.bak>\n"
            )

    failures = []
    orig_dirs = sp.ppu_sdk_include_dirs
    try:
        # wt_v3 layout: only the hggc_* spellings exist.
        sp.ppu_sdk_include_dirs = lambda: [sdk_inc]
        sp._PREFIXED_INCLUDE_MAP = {}
        for name in ("hggc_vector_types.h", "hggc_driver_types.h",
                     "hggc_vector_functions.h"):
            open(os.path.join(sdk_inc, name), "w").close()
        sp._enable_prefixed_core_includes()
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        for want in ("<hggc_vector_types.h>", "<hggc_driver_types.h>",
                     "<some_dir/hggc_vector_functions.h>"):
            if want not in text:
                failures.append("prefixed include rewrite missing: " + want)
        if "<vector_types.h>" in text:
            failures.append("unprefixed vector_types.h survived")
        if "<vector_types.h.bak>" not in text:
            failures.append("rewrite touched a longer filename")

        # Flat layout: the unprefixed CUDA names exist, nothing rewrites.
        open(os.path.join(sdk_inc, "vector_types.h"), "w").close()
        sp._PREFIXED_INCLUDE_MAP = {}
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        if "<vector_types.h>" not in text:
            failures.append("flat-layout SDK was rewritten anyway")
    finally:
        sp.ppu_sdk_include_dirs = orig_dirs
        sp._PREFIXED_INCLUDE_MAP = {}
    return failures


def test_sdk22_state_resolved_with_compat_disabled(tmp):
    import sailify.sailify_python as sp

    src_dir = os.path.join(tmp, "compat_off")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "k.cu")

    def write_case():
        with open(src, "w") as fh:
            fh.write(
                "int run(int* p) { __nv_atomic_add(p, 1, 0, "
                "__NV_THREAD_SCOPE_SYSTEM); return 0; }\n"
            )

    failures = []
    orig_detect = sp.detect_ppu_sdk_version
    try:
        sp.detect_ppu_sdk_version = lambda: (2, 1, 5)
        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        if "__hg_atomic_add" not in text:
            failures.append("__nv_atomic_add mapping is not unconditional")
        if "__HG_THREAD_SCOPE_SYSTEM" in text:
            failures.append("stale SDK 2.2 state leaked into sailify_extra_files")

        sp._maybe_enable_sdk_2_2_mappings(None)
        sp.detect_ppu_sdk_version = lambda: (2, 2, 0)
        write_case()
        sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(src) as fh:
            text = fh.read()
        if "__hg_atomic_add" not in text:
            failures.append("__nv_atomic_add mapping is not unconditional")
        if "__NV_THREAD_SCOPE_SYSTEM" in text or "__HG_THREAD_SCOPE_SYSTEM" not in text:
            failures.append("sailify_extra_files did not apply SDK 2.2 mappings")
    finally:
        sp.detect_ppu_sdk_version = orig_detect
        sp._maybe_enable_sdk_2_2_mappings(None)
    return failures


def test_nvcc_flag_mapping():
    out = translate([
        "-c", "-O2", "-std=c++17", "-t=0",
        "--expt-relaxed-constexpr", "--use_fast_math",
        "--ptxas-options", "-warn-spills",
        "--source-in-ptx",
        "-Xcompiler", "-O3",
    ])
    failures = []
    if "--llvm-options" not in out:
        failures.append("--ptxas-options not mapped to --llvm-options")
    idx = out.index("--llvm-options") if "--llvm-options" in out else -1
    if idx >= 0 and (idx + 1 >= len(out) or out[idx + 1] != "-ppu-warn-spills"):
        failures.append("--ptxas-options value not remapped to -ppu-warn-spills")
    if "--source-in-ptx" in out:
        failures.append("--source-in-ptx not dropped")
    for passthrough in ("-c", "-O2", "-std=c++17", "-t=0",
                        "--expt-relaxed-constexpr", "--use_fast_math",
                        "-Xcompiler", "-O3"):
        if passthrough not in out:
            failures.append(passthrough + " should pass through unchanged")
    out2 = translate(["--ptxas-options=--warn-on-spills", "--resource-usage", "--keep"])
    if out2 != ["--ptxas-options=--warn-on-spills", "--resource-usage", "--keep"]:
        failures.append("unrecognised flags should pass through unchanged: " + str(out2))
    return failures


def test_sail_ns_root_only_rewrite():
    """Step-2c rewrites only ROOT cccl references; nested qualifications
    (cuda::cub, at_cuda_detail::cub) are project knowledge and stay put."""
    import sailify.sailify_python as sp

    sdk_inc = tempfile.mkdtemp(prefix="sailify_ns_sdk_")
    for fam in ("cub", "thrust", "cooperative_groups"):
        os.makedirs(os.path.join(sdk_inc, "sail", fam), exist_ok=True)

    failures = []
    orig_dirs = sp.ppu_sdk_include_dirs
    try:
        sp.ppu_sdk_include_dirs = lambda: [sdk_inc]
        sp._SAIL_INCLUDE_FAMILIES = ()
        sp._SAIL_NS_COMBINED_RE = None
        sp._enable_sail_include_prefixes()
        if sp._SAIL_NS_COMBINED_RE is None:
            return ["sail/ nested SDK did not activate the ns rewrite"]

        def rewrite(text):
            # all-spaces mask = apply everywhere (the caller masks
            # comments/strings/includes; this test targets the regex itself)
            return sp._sub_masked(
                sp._SAIL_NS_COMBINED_RE, sp._sail_ns_repl, text, " " * len(text)
            )

        # root references get rewritten, with a leading ::
        for src, want in (
            ("cub::DeviceScan::SortPairs",
             "::sail::cub::DeviceScan::SortPairs"),
            ("::cub::Max()", "::sail::cub::Max()"),
            ("auto v = ::cub::DeviceReduce::Sum;",
             "auto v = ::sail::cub::DeviceReduce::Sum;"),
            ("ROCM_HIPCUB(::cub)::Max()",
             "ROCM_HIPCUB(::sail::cub)::Max()"),
            ("namespace cub {", "namespace sail::cub {"),
            ("using namespace cub;", "using namespace ::sail::cub;"),
            ("namespace cg = cooperative_groups;",
             "namespace cg = ::sail::cooperative_groups;"),
            ("thrust::device_vector<int> v;",
             "::sail::thrust::device_vector<int> v;"),
        ):
            got = rewrite(src)
            if got != want:
                failures.append("root rewrite %r -> %r (want %r)" % (src, got, want))

        # nested qualifications are left alone: whether the enclosing
        # namespace is a CUB_WRAPPED_NAMESPACE-style project namespace or a
        # persona namespace the main conversion renames is not this step's
        # to decide.
        for keep in (
            "namespace at::cuda::cub {",
            "at_cuda_detail::cub::DeviceScan::InclusiveSum",
            "ROCM_HIPCUB(at_cuda_detail::cub)::BlockLoad",
            "namespace cub = ::at_cuda_detail::cub;",
            "cuda::cub::x",
            "int cub = 3;",
            "for (int cub = 0; cub < n; ++cub)",
        ):
            got = rewrite(keep)
            if got != keep:
                failures.append("nested/plain %r rewritten to %r" % (keep, got))

        # idempotent on already-rewritten output
        once = rewrite("::cub::Max() ::thrust::foo namespace sail::cub {")
        if rewrite(once) != once or "::sail::sail" in once:
            failures.append("rewrite not idempotent: %r" % once)

        # flat-layout SDK (no sail/ nesting): no rewrite machinery at all
        flat_inc = tempfile.mkdtemp(prefix="sailify_ns_flat_")
        sp.ppu_sdk_include_dirs = lambda: [flat_inc]
        sp._SAIL_INCLUDE_FAMILIES = ()
        sp._SAIL_NS_COMBINED_RE = None
        sp._enable_sail_include_prefixes()
        if sp._SAIL_NS_COMBINED_RE is not None:
            failures.append("flat-layout SDK activated the ns rewrite")
    finally:
        sp.ppu_sdk_include_dirs = orig_dirs
        sp._SAIL_INCLUDE_FAMILIES = ()
        sp._SAIL_NS_COMBINED_RE = None
    return failures


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="sailify_api_patterns_")
    npass = 0
    nfail = 0
    try:
        for name, fn in [
            ("pytorch_codegen_string_conversion", test_pytorch_codegen_string_conversion),
            ("pytorch_write_compat_wrapper_header", lambda: test_write_compat_wrapper_header(tmp)),
            ("sglang_jit_mirror_inplace", lambda: test_sglang_jit_mirror(tmp)),
            ("pytorch_cpp_extension_jit", lambda: test_pytorch_cpp_extension_jit(tmp)),
            ("sglang_aot_preprocessor_direct", lambda: test_sglang_aot_preprocessor_direct(tmp)),
            ("sdk22_mapping_state_toggles_with_version", test_sdk22_mapping_state_toggles_with_version),
            ("sdk22_bf16_pipeline_spelling_rewrite", lambda: test_sdk22_bf16_pipeline_spelling_rewrite(tmp)),
            ("prefixed_core_include_probe", lambda: test_prefixed_core_include_probe(tmp)),
            ("sdk22_state_resolved_with_compat_disabled", lambda: test_sdk22_state_resolved_with_compat_disabled(tmp)),
            ("nvcc_flag_mapping", test_nvcc_flag_mapping),
            ("sail_ns_root_only_rewrite", test_sail_ns_root_only_rewrite),
        ]:
            try:
                failures = fn()
            except Exception as exc:
                failures = ["exception: %r" % (exc,)]
            if failures:
                nfail += 1
                for f in failures:
                    print("FAIL api_pattern " + name + ": " + f)
            else:
                npass += 1
                print("PASS api_pattern " + name)
    finally:
        if args.keep:
            print("artifacts kept under: " + tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return _report(npass, nfail)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

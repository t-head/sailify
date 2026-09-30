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

    failures = []
    try:
        sp._maybe_enable_sdk_2_2_mappings((2, 2, 0))
        if not sp._SDK22_MAP or sp._SDK22_RE is None:
            failures.append("SDK 2.2 mappings not enabled for version (2, 2, 0)")
        elif sp._SDK22_RE.search("__NV_THREAD_SCOPE_THREAD") is None:
            failures.append("SDK 2.2 regex does not match a mapped identifier")

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
        sp._maybe_enable_sdk_2_2_mappings(None)
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
            ("sdk22_state_resolved_with_compat_disabled", lambda: test_sdk22_state_resolved_with_compat_disabled(tmp)),
            ("nvcc_flag_mapping", test_nvcc_flag_mapping),
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

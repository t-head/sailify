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


_RERUN_KERNEL_CU = """#include "rerun_util.cuh"

__global__ void k(float* out) {
    rerun_scale(out, 2.0f);
}
"""

_RERUN_UTIL_CUH = """#pragma once
__device__ __forceinline__ void rerun_scale(float* p, float v) {
    *p = v;
}
"""


def test_jit_rerun_inject_recursion(tmp):
    from sailify.sailify_python import sailify_extra_files_recursive

    root = os.path.join(tmp, "jit_rerun")
    build = os.path.join(root, "build")
    os.makedirs(build, exist_ok=True)
    src = os.path.join(build, "kernel.cu")
    hdr = os.path.join(build, "rerun_util.cuh")
    with open(src, "w") as fh:
        fh.write(_RERUN_KERNEL_CU)
    with open(hdr, "w") as fh:
        fh.write(_RERUN_UTIL_CUH)
    compat = os.path.join(build, ".ppu_compat", "compatible_wrapper.h")
    inject_line = "#include <.ppu_compat/compatible_wrapper.h>"

    failures = []
    for _ in range(2):
        results = sailify_extra_files_recursive(
            output_directory=build,
            extra_files=[src],
            header_include_dirs=[],
            inject_compat_include=True,
        )
        if compat in [os.path.abspath(p) for p in results]:
            failures.append("compatible_wrapper.h entered the BFS queue")
        if not os.path.isfile(compat):
            failures.append("compatible_wrapper.h not generated")
            return failures
        with open(compat) as fh:
            first = fh.read().splitlines()[0]
        if first.strip() == inject_line:
            failures.append("compatible_wrapper.h re-injected into itself")
        with open(hdr) as fh:
            hdr_first = fh.read().splitlines()[0]
        if hdr_first.strip() != inject_line:
            failures.append("rerun_util.cuh missing injected wrapper include")

    wheel = os.path.join(root, "wheel_inc")
    os.makedirs(os.path.join(wheel, ".ppu_compat"), exist_ok=True)
    wheel_compat = os.path.join(wheel, ".ppu_compat", "compatible_wrapper.h")
    with open(wheel_compat, "w") as fh:
        fh.write("#ifndef COMPATIBLE_WRAPPER_H\n#define COMPATIBLE_WRAPPER_H\n#endif\n")
    with open(os.path.join(wheel, "wheel_util.cuh"), "w") as fh:
        fh.write(inject_line + "\n#pragma once\n")
    build2 = os.path.join(root, "build2")
    os.makedirs(build2, exist_ok=True)
    src2 = os.path.join(build2, "kernel.cu")
    with open(src2, "w") as fh:
        fh.write('#include <wheel_util.cuh>\n__global__ void k2() {}\n')

    results = sailify_extra_files_recursive(
        output_directory=build2,
        extra_files=[src2],
        header_include_dirs=[wheel],
        inject_compat_include=True,
    )
    if wheel_compat in [os.path.abspath(p) for p in results]:
        failures.append("wheel compatible_wrapper.h entered the BFS queue")
    with open(wheel_compat) as fh:
        first = fh.read().splitlines()[0]
    if first.strip() == inject_line:
        failures.append("wheel compatible_wrapper.h rewritten via header_include_dirs")

    miss_dir = os.path.join(root, "miss", ".ppu_compat")
    hit_dir = os.path.join(root, "hit")
    os.makedirs(miss_dir, exist_ok=True)
    os.makedirs(hit_dir, exist_ok=True)
    late_hdr = os.path.join(hit_dir, "late_util.cuh")
    with open(late_hdr, "w") as fh:
        fh.write(_RERUN_UTIL_CUH.replace("rerun_scale", "late_scale"))
    build3 = os.path.join(root, "build3")
    os.makedirs(build3, exist_ok=True)
    src3 = os.path.join(build3, "kernel.cu")
    with open(src3, "w") as fh:
        fh.write("#include <late_util.cuh>\n__global__ void k3() {}\n")
    results = sailify_extra_files_recursive(
        output_directory=build3,
        extra_files=[src3],
        header_include_dirs=[miss_dir, hit_dir],
        inject_compat_include=True,
    )
    if os.path.abspath(late_hdr) not in results:
        failures.append("header in later include dir not discovered past .ppu_compat")
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


_SDK23_RENAME_CASES = [
    ("#include <channel_descriptor.h>", "#include <hggc_channel_descriptor.h>"),
    ("#include <driver_functions.h>", "#include <hggc_driver_functions.h>"),
    ("#include <driver_types.h>", "#include <hggc_driver_types.h>"),
    ("#include <host_defines.h>", "#include <hggc_host_defines.h>"),
    ("#include <vector_functions.h>", "#include <hggc_vector_functions.h>"),
    ("#include <vector_functions.hpp>", "#include <hggc_vector_functions.hpp>"),
    ("#include <vector_types.h>", "#include <hggc_vector_types.h>"),
    ('#include "texture_types.h.h"', '#include "hggc_texture_types.h"'),
    ('#include "texture_types.h"', '#include "hggc_texture_types.hpp"'),
    ('#include "sub/vector_types.h"', '#include "sub/hggc_vector_types.h"'),
    ("#include <cuda_runtime.h>", "#include <hggc_runtime.h>"),
    ("#include<vector_types.h>", "#include<hggc_vector_types.h>"),
    ('#include"host_defines.h"', '#include"hggc_host_defines.h"'),
    ("# include <vector_types.h>", "# include <hggc_vector_types.h>"),
]


def test_sdk23_renames_toggles_with_version():
    import sailify.sailify_python as sp

    failures = []
    try:
        sp._maybe_enable_sdk_version_mappings((2, 3, 0))
        if not sp._SDK23_RENAME_ACTIVE:
            failures.append("SDK 2.3 renames not enabled for version (2, 3, 0)")
        for src, expected in _SDK23_RENAME_CASES:
            got = sp._rewrite_include_directives(src)
            if got != expected:
                failures.append("SDK 2.3 rename %r -> %r, expected %r" % (src, got, expected))

        for version in ((2, 2, 9), (2, 2, 2), (2, 2, 1), (2, 2, 0), (2, 1, 5), None):
            sp._maybe_enable_sdk_version_mappings(version)
            if sp._SDK23_RENAME_ACTIVE:
                failures.append("SDK 2.3 renames not disabled for version %r" % (version,))
            for src, expected in _SDK23_RENAME_CASES[:9]:
                got = sp._rewrite_include_directives(src)
                if "hggc_" in got:
                    failures.append("legacy include rewritten for version %r: %r -> %r" % (version, src, got))
                    break

        sp._maybe_enable_sdk_version_mappings((2, 3, 1))
        if not sp._SDK23_RENAME_ACTIVE:
            failures.append("SDK 2.3 renames not enabled for version (2, 3, 1)")
    finally:
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk222_device_function_mapping(tmp):
    import sailify.sailify_python as sp

    cases = [
        ("float e = __nv_erff(x);", "float e = __hg_erff(x);"),
        ("float c = __nv_erfcf(x);", "float c = __hg_erfcf(x);"),
        ("float i = __nv_erfcinvf(y);", "float i = __hg_erfcinvf(y);"),
        ("float c = __nv_erfcxf(x);", "float c = __hg_erfcxf(x);"),
        ("float i = __nv_erfinvf(y);", "float i = __hg_erfinvf(y);"),
        ("double e = __nv_erf(x);", "double e = __hg_erf(x);"),
        ("double c = __nv_erfc(x);", "double c = __hg_erfc(x);"),
        ("double i = __nv_erfcinv(y);", "double i = __hg_erfcinv(y);"),
        ("double c = __nv_erfcx(x);", "double c = __hg_erfcx(x);"),
        ("double i = __nv_erfinv(y);", "double i = __hg_erfinv(y);"),
        ("__nv_erff(x) + __nv_erf(y)", "__hg_erff(x) + __hg_erf(y)"),
        ('const char* code = "__nv_erff(x)";', 'const char* code = "__hg_erff(x)";'),
    ]
    untouched = "\n".join([
        '#include "__nv_erff.h"',
        '# include "detail/__nv_erfcinv.h"',
        '#include <__nv_erfinvf.h>',
        '#include "__NV_THREAD_SCOPE_SYSTEM.h"',
        "// __nv_erff(x)",
        "/* __nv_erfc(x) */",
        "int my__nv_erff_var = 0;",
        "__nv_erfcinvf_suffix(x);",
        "prefix__nv_erfinv(x);",
        "float e = erff(x);",
        "double e = erf(x);",
        "float e = __hg_erff(x);",
        "float s = __nv_sqrtf(x);",
        "float f = __nv_fmaf(x, y, z);",
    ]) + "\n"
    source = "\n".join(src for src, _ in cases) + "\n" + untouched
    converted = "\n".join(dst for _, dst in cases) + "\n" + untouched
    versions = [
        ((2, 2, 2), True),
        ((2, 2, 1), False),
        ((2, 2, 3), True),
        (None, False),
        ((2, 2, 99), True),
        ((2, 2, 0), False),
        ((2, 3, 0), True),
        ((2, 1, 5), False),
        ((2, 3, 1), True),
        ((2, 4, 0), True),
        ((3, 0, 0), True),
    ]
    src_dir = os.path.join(tmp, "sdk222_device_functions")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "error_functions.cu")
    failures = []
    orig_detect = sp.detect_ppu_sdk_version
    try:
        for entry in ("project", "extra_files"):
            for version, enabled in versions:
                sp.detect_ppu_sdk_version = lambda: version
                with open(src, "w") as fh:
                    fh.write(source)
                expected = converted if enabled else source
                for iteration in range(2):
                    if entry == "project":
                        results = sp.sailify(
                            src_dir, ppu_sdk_version=version, install_ppu_compat=False,
                        )
                    else:
                        results = sp.sailify_extra_files(
                            src_dir, [src], install_ppu_compat=False,
                        )
                    with open(src) as fh:
                        got = fh.read()
                    if src not in results or results[src].status != "ok" or got != expected:
                        failures.append(
                            "%s SDK %r iteration %d: expected %r, got %r"
                            % (entry, version, iteration, expected, got)
                        )
    finally:
        sp.detect_ppu_sdk_version = orig_detect
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk222_lexical_boundaries(tmp):
    import sailify.sailify_python as sp

    call = "float e = __nv_erff(x);\n"
    mapped = call.replace("__nv_erff", "__hg_erff")
    cases = []
    for directive in (
        '#include /* libdevice */ "__nv_erff.h"',
        '#include /*\n*/ "__nv_erff.h"',
        '/*\n*/ #include "__nv_erff.h"',
        '#include /\\\n* SDK *\\\n/ "__nv_erff.h"',
        '# /* before */ include /* after */ <__nv_erff.h>',
        '/* leading */ #include "__nv_erff.h"',
        '#include \\\n "__nv_erff.h"',
        '#\\\ninc\\\nlude /* path */ \\\n <__nv_erff.h>',
        '#include "detail/\\\n__nv_erff.h"',
        '#include "detail//__nv_erff.h"',
        '#include "detail/1\'000/__nv_erff.h"',
    ):
        cases.append((directive + "\n" + call, directive + "\n" + mapped))
    for newline in ("\n", "\r\n"):
        for backslashes in ("\\", "\\\\"):
            comment = "// continued " + backslashes + newline + "__nv_erff(x)" + newline
            cases.append((comment + call, comment + mapped))
    for number in (
        "1'000", "0xDEAD'BEEF", "0b10'01", "1.2'5e+1'0", ".1'25", "1'000_km",
        "1\\\n'000", "1'\\\n000", ".\\\n1'25", "1.2e\\\n+1'0",
    ):
        declaration = "auto n = " + number + ";\n"
        cases.append((declaration + call, declaration + mapped))
    for literal in ("'" + "\\" * 2 + "\n''", "'" + "\\" * 2 + "\n'__nv_erff'"):
        declaration = "int c = " + literal + ";\n"
        cases.append((declaration + call, declaration + mapped))
    cases.extend([
        ('/\\\n/ __nv_erff(x)\n' + call, '/\\\n/ __nv_erff(x)\n' + mapped),
        ('/\\\n* __nv_erff(x) *\\\n/\n' + call, '/\\\n* __nv_erff(x) *\\\n/\n' + mapped),
        ('#define H <stddef.h>\n#include /**/ H\n// /**/ "cudaMalloc"\n' + call,
         '#define H <stddef.h>\n#include /**/ H\n// /**/ "cudaMalloc"\n' + mapped),
        ('#define H <stddef.h>\n#include /**/ H\nconst char* s = /**/ "__nv_erff(x)";\n',
         '#define H <stddef.h>\n#include /**/ H\nconst char* s = /**/ "__hg_erff(x)";\n'),
        ('#define CODE /*\n*/ #include "__nv_erff.h"\n',
         '#define CODE /*\n*/ #include "__hg_erff.h"\n'),
        ('const char* s = "' + "\\" * 2 + '\n"__nv_erff";\n' + call,
         'const char* s = "' + "\\" * 2 + '\n"__hg_erff";\n' + mapped),
        ("char c = 'x'; char d = '\\''; auto u = u8'a';\n" + call,
         "char c = 'x'; char d = '\\''; auto u = u8'a';\n" + mapped),
        ('const char* code = "__nv_erff(x)";\n',
         'const char* code = "__hg_erff(x)";\n'),
        ('const char* code = R"src(\n#include "__nv_erff.h"\n__nv_erff(x)\n)src";\n',
         'const char* code = R"src(\n#include "__hg_erff.h"\n__hg_erff(x)\n)src";\n'),
        ("int n = 1'000; cudaMalloc(&p, n);\n" + call,
         "int n = 1'000; hggcMalloc(&p, n);\n" + mapped),
    ])
    root = os.path.join(tmp, "lexical")
    os.makedirs(root, exist_ok=True)
    src = os.path.join(root, "kernel.cu")
    failures = []
    try:
        for index, (source, expected) in enumerate(cases):
            with open(src, "w", newline="") as fh:
                fh.write(source)
            results = sp.sailify(root, ppu_sdk_version=(2, 2, 2), install_ppu_compat=False)
            with open(src) as fh:
                actual = fh.read()
            if results[src].status != "ok" or actual != expected.replace("\r\n", "\n"):
                failures.append("lexical case %d: %r" % (index, actual))
            masked = sp._mask_comments_and_strings(source, mask_includes=True)
            if len(masked) != len(source):
                failures.append("mask length differs in case %d" % index)
    finally:
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk222_concurrent_conversion(tmp):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from unittest.mock import patch
    import sailify.sailify_python as sp

    barrier = Barrier(2)
    preprocessor = sp.preprocessor
    source = "float e = __nv_erff(x);\n"

    def process(**kwargs):
        barrier.wait(timeout=10)
        return preprocessor(**kwargs)

    def convert(version):
        root = os.path.join(tmp, "concurrent", str(version))
        os.makedirs(root, exist_ok=True)
        src = os.path.join(root, "kernel.cu")
        with open(src, "w") as fh:
            fh.write(source)
        results = sp.sailify(root, ppu_sdk_version=version, install_ppu_compat=False)
        with open(src) as fh:
            return results[src].status, fh.read()

    with patch.object(sp, "preprocessor", side_effect=process):
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(convert, [(2, 2, 2), (2, 2, 1)]))
    expected = [("ok", source.replace("__nv_", "__hg_")), ("ok", source)]
    return [] if actual == expected else ["concurrent SDK selection: %r" % (actual,)]


def test_sdk222_detected_compiler_switch(tmp):
    from unittest.mock import patch
    import sailify.sailify_python as sp

    source = "float e = __nv_erff(x);\n"
    roots = []
    for version in ("2.2.2", "2.2.1"):
        root = os.path.join(tmp, "detector", version)
        os.makedirs(os.path.join(root, "bin"), exist_ok=True)
        compiler = os.path.join(root, "bin", "hgcc")
        with open(compiler, "w") as fh:
            fh.write("#!" + sys.executable + "\nprint('Release version " + version + "')\n")
        os.chmod(compiler, 0o755)
        roots.append(root)

    src = os.path.join(tmp, "detected.cu")
    failures = []
    try:
        for entry in ("direct", "extra_files", "recursive"):
            for env_var in ("PATH", "PPU_SDK"):
                for index in (0, 1, 0):
                    env = {"PATH": "", "PPU_SDK": "", "PPU_HOME": ""}
                    env[env_var] = os.path.join(roots[index], "bin") if env_var == "PATH" else roots[index]
                    with patch.dict(os.environ, env):
                        with open(src, "w") as fh:
                            fh.write(source)
                        if entry == "direct":
                            result = sp.preprocessor(
                                output_directory=tmp, filepath=src, all_files=[src],
                                stats={}, clean_ctx=None, show_progress=False,
                                project_directory=tmp,
                            )
                        elif entry == "recursive":
                            result = sp.sailify_extra_files_recursive(
                                tmp, [src], install_ppu_compat=False,
                            )[src]
                        else:
                            result = sp.sailify_extra_files(tmp, [src], install_ppu_compat=False)[src]
                        with open(src) as fh:
                            actual = fh.read()
                    expected = source.replace("__nv_", "__hg_") if index == 0 else source
                    if result.status != "ok" or actual != expected:
                        failures.append("%s compiler switch via %s to %s: %r" % (entry, env_var, roots[index], actual))
    finally:
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk222_explicit_entry_version(tmp):
    from unittest.mock import patch
    import sailify.sailify_python as sp

    root = os.path.join(tmp, "explicit_sdk")
    os.makedirs(root, exist_ok=True)
    src = os.path.join(root, "kernel.cu")
    source = "float e = __nv_erff(x);\n"
    failures = []
    try:
        for entry in ("direct", "project"):
            for selected, detected in (((2, 2, 2), (2, 2, 1)), ((2, 2, 1), (2, 2, 2)), (None, None)):
                with open(src, "w") as fh:
                    fh.write(source)
                with patch.object(sp, "detect_ppu_sdk_version", return_value=detected) as detect:
                    if entry == "direct":
                        sp.preprocessor(
                            output_directory=root, filepath=src, all_files=[src],
                            stats={}, clean_ctx=None, show_progress=False,
                            project_directory=root, ppu_sdk_version=selected,
                        )
                    else:
                        sp.sailify(root, ppu_sdk_version=selected, install_ppu_compat=False)
                    expected_calls = int(entry == "project" and selected is None)
                    if detect.call_count != expected_calls:
                        failures.append("%s re-detected explicit SDK %r" % (entry, selected))
                with open(src) as fh:
                    actual = fh.read()
                expected = source.replace("__nv_", "__hg_") if selected == (2, 2, 2) else source
                if actual != expected:
                    failures.append("%s explicit SDK %r: %r" % (entry, selected, actual))
    finally:
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk222_compiler_identity(tmp):
    from unittest.mock import patch
    import sailify.ppu_sdk_version as sv
    import sailify.sailify_python as sp

    root = os.path.join(tmp, "compiler_identity")
    os.makedirs(root, exist_ok=True)
    calls_path = os.path.join(root, "calls.txt")
    src = os.path.join(root, "kernel.cu")
    source = "float e = __nv_erff(x);\n"
    failures = []

    def write_compiler(path, sdk, runtime):
        content = (
            "#!" + sys.executable + "\nimport sys\n"
            + "with open(" + repr(calls_path) + ", 'a') as fh:\n    fh.write(sys.argv[0] + '\\n')\n"
            + "print('Release version " + sdk + "')\n"
            + "print('Runtime API version v" + str(runtime) + "')\n"
        )
        with open(path, "w") as fh:
            fh.write(content)
        os.chmod(path, 0o755)

    def calls():
        with open(calls_path) as fh:
            return fh.read().splitlines()

    compilers = []
    for sdk, runtime in (("2.2.2", 3), ("2.2.1", 2)):
        path = os.path.join(root, "hgcc-" + sdk)
        write_compiler(path, sdk, runtime)
        compilers.append(path)

    try:
        for mode in ("symlink", "inplace"):
            home = os.path.join(root, mode)
            os.makedirs(os.path.join(home, "bin"), exist_ok=True)
            path = os.path.join(home, "bin", "hgcc")
            for index in (0, 1, 0):
                sdk = (2, 2, 2) if index == 0 else (2, 2, 1)
                runtime = 3 if index == 0 else 2
                if mode == "symlink":
                    if os.path.islink(path):
                        os.unlink(path)
                    os.symlink(compilers[index], path)
                else:
                    previous = os.stat(path) if os.path.exists(path) else None
                    write_compiler(path, ".".join(map(str, sdk)), runtime)
                    if previous is not None:
                        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                env = {"PATH": "", "PPU_SDK": home, "PPU_HOME": ""}
                with patch.dict(os.environ, env):
                    actual_sdk = sv.detect_ppu_sdk_version()
                    first_calls = calls()
                    actual_runtime = sv.detect_runtime_api_version()
                    with open(src, "w") as fh:
                        fh.write(source)
                    result = sp.sailify_extra_files(root, [src], install_ppu_compat=False)[src]
                    with open(src) as fh:
                        actual = fh.read()
                expected = source.replace("__nv_", "__hg_") if index == 0 else source
                if (actual_sdk, actual_runtime, result.status, actual) != (sdk, runtime, "ok", expected):
                    failures.append("%s compiler change: %r" % (mode, (actual_sdk, actual_runtime, actual)))
                if calls() != first_calls:
                    failures.append("%s unchanged compiler missed cache" % mode)
                if first_calls[-1] != path:
                    failures.append("%s invoked resolved path instead of original path" % mode)
    finally:
        sp._maybe_enable_sdk_version_mappings(None)
    return failures


def test_sdk23_compat_shims(tmp):
    import sailify.sailify_python as sp

    failures = []
    compat = os.path.join(tmp, "shims", ".ppu_compat")
    sp.write_compat_wrapper_header(compat, ppu_sdk_version=(2, 3, 0))
    for name, fallback in sp._SDK23_SHIM_HEADERS:
        path = os.path.join(compat, name)
        if not os.path.isfile(path):
            failures.append("missing .ppu_compat shim " + name)
            continue
        with open(path) as f:
            content = f.read()
        if f"#include_next <{name}>" not in content:
            failures.append("shim %s has no include_next pass-through" % name)
        if f"#include <{fallback}>" not in content:
            failures.append("shim %s does not forward to %s" % (name, fallback))
        if "#ifndef" in content or "#define PPU" in content:
            failures.append("shim %s contains a guard" % name)

    for idx, version in enumerate(((2, 2, 2), (2, 2, 1), (2, 2, 0), (2, 1, 5), None)):
        compat = os.path.join(tmp, "shims", str(idx))
        sp.write_compat_wrapper_header(compat, ppu_sdk_version=version)
        for name, _ in sp._SDK23_SHIM_HEADERS:
            if os.path.isfile(os.path.join(compat, name)):
                failures.append("shim %s generated for SDK version %r" % (name, version))
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


_RT_VERSIONED_CASES = [
    ("cudaStreamGetCaptureInfo_v2(stream, &st, &id);", "hggcStreamGetCaptureInfo_v2("),
    ("cudaStreamGetCaptureInfo_v3(stream, &st, &id, &g, &deps, &edges, &n);", "hggcStreamGetCaptureInfo_v3("),
    ("cudaGraphGetEdges_v2(g, from, to, ed, &n);", "hggcGraphGetEdges_v2("),
    ("cudaGraphNodeGetDependencies_v2(node, deps, ed, &n);", "hggcGraphNodeGetDependencies_v2("),
    ("cudaGraphNodeGetDependentNodes_v2(node, deps, ed, &n);", "hggcGraphNodeGetDependentNodes_v2("),
    ("cudaGraphAddDependencies_v2(g, from, to, ed, n);", "hggcGraphAddDependencies_v2("),
    ("cudaGraphRemoveDependencies_v2(g, from, to, ed, n);", "hggcGraphRemoveDependencies_v2("),
    ("cudaGraphAddNode_v2(&node, g, deps, ed, n, &params);", "hggcGraphAddNode_v2("),
    ("cudaGetDeviceProperties_v2(&prop, 0);", "hggcGetDeviceProperties_v2("),
    ("cudaEventElapsedTime_v2(&ms, a, b);", "hggcEventElapsedTime_v2("),
    ("cudaMemPrefetchAsync_v2(p, n, loc, 0, stream);", "hggcMemPrefetchAsync_v2("),
    ("cudaMemAdvise_v2(p, n, advice, loc);", "hggcMemAdvise_v2("),
    ("cudaStreamUpdateCaptureDependencies_v2(stream, deps, ed, n, cudaStreamSetCaptureDependencies);",
     "hggcStreamUpdateCaptureDependencies_v2("),
    ("cudaStreamUpdateCaptureDependencies_v2(stream, deps, ed, n, cudaStreamSetCaptureDependencies);",
     "hggcStreamSetCaptureDependencies)"),
    ("params.type = cudaGraphNodeTypeConditional;", "hggcGraphNodeTypeConditional"),
    ("params.conditional.type = cudaGraphCondTypeIf;", "hggcGraphCondTypeIf"),
    ("cudaSignalExternalSemaphoresAsync_v2(sems, params, n, stream);", "hggcSignalExternalSemaphoresAsync_v2("),
    ("cudaWaitExternalSemaphoresAsync_v2(sems, params, n, stream);", "hggcWaitExternalSemaphoresAsync_v2("),
    ("cudaLaunchHostFunc_v2(stream, fn, data, 0);", "hggcLaunchHostFunc_v2("),
    ("cudaGraphConditionalHandleCreate_v2(&h, g, ctx, 0, 0);", "hggcGraphConditionalHandleCreate_v2("),
    ("desc.f = cudaChannelFormatKindFloat;", "hggcChannelFormatKindFloat"),
    ("cudaJitOption opt = cudaJitMaxRegisters;", "hggcJitMaxRegisters"),
    ("cudaDeviceGetAttribute(&v, cudaDevAttrMaxTexture2DWidth, 0);", "hggcDevAttrMaxTexture2DWidth"),
    ("attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;", "HG_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION"),
    ("cudaKernelNodeParamsV2 p2; cudaMemcpyNodeParams m2;", "hggcKernelNodeParamsV2"),
    ("cudaGreenCtxCreate(&gctx, 0);", "hggcGreenCtxCreate("),
    ("cuLogicalEndpointCreate(&ep, 0, 0);", "hgLogicalEndpointCreate("),
    ("unsigned int t = cudaTextureType2D;", "hggcTextureType2D"),
]

_RT_DRIVER_CASES = [
    ("cuLaunchHostFunc_v2(ctx, stream, fn, data, 0);", "hgLaunchHostFunc_v2("),
    ("cuMulticastBindMem_v2(mc, dptr, 0, 0);", "hgMulticastBindMem_v2("),
    ("cuMulticastBindAddr_v2(mc, &addr, 0);", "hgMulticastBindAddr_v2("),
]


def _rt_convert(sp, tmp, text, name):
    src = os.path.join(tmp, name)
    with open(src, "w") as fh:
        fh.write(text + "\n")
    res = sp.preprocessor(
        output_directory=tmp,
        filepath=src,
        all_files=[],
        stats={},
        clean_ctx=None,
        show_progress=False,
        project_directory=tmp,
    )
    if res.status != "ok":
        return None
    with open(src) as fh:
        return fh.read()


def test_versioned_runtime_api_conversion(tmp):
    import sailify.sailify_python as sp

    failures = []
    for i, (src, expected) in enumerate(_RT_VERSIONED_CASES):
        got = _rt_convert(sp, tmp, src, "rt_ver_%d.cu" % i)
        if got is None:
            failures.append("preprocessor failed for: " + src)
        elif expected not in got:
            failures.append(src + " did not produce " + expected + ": " + got.strip())
    for i, (src, expected) in enumerate(_RT_DRIVER_CASES):
        got = _rt_convert(sp, tmp, src, "rt_drv_%d.cu" % i)
        if got is None or expected not in got:
            failures.append(src + " did not produce " + expected)
    return failures


def test_nvcc_flag_mapping(tmp):
    from sailify import translate_nvcc_options
    failures = []

    def expect(name, actual, wanted):
        if actual != wanted:
            failures.append(name + ": got " + str(actual) + ", want " + str(wanted))

    # passthrough + classic separated/short forms
    expect(
        "classic passthrough",
        translate([
            "-c", "-O2", "-std=c++17", "-t=0",
            "--expt-relaxed-constexpr", "--use_fast_math",
            "--ptxas-options", "-warn-spills",
            "--source-in-ptx",
            "-Xcompiler", "-O3",
        ]),
        ["-c", "-O2", "-std=c++17", "-t=0",
         "--expt-relaxed-constexpr", "--use_fast_math",
         "-Xcompiler", "-O3",
         "--llvm-options", "-ppu-warn-spills"],
    )

    # short form with a single value
    expect(
        "short -Xptxas single value",
        translate_nvcc_options(["-Xptxas", "-v"]),
        ["--llvm-options", "--resource-usage"],
    )

    # joined '=' form whose value itself contains '=' (split at first '=')
    expect(
        "joined unknown value dropped",
        translate_nvcc_options(["-Xptxas=--split-compile=0", "--keep"]),
        ["--keep"],
    )

    # comma-delimited value list: recognised element translated, unknown dropped
    expect(
        "comma list mixed known/unknown",
        translate_nvcc_options(["--ptxas-options=-v,--register-usage-level=10"]),
        ["--llvm-options", "--resource-usage"],
    )
    expect(
        "comma list two known values",
        translate_nvcc_options(["-Xptxas=-v,--def-load-cache=ca"]),
        ["--llvm-options", "--resource-usage", "--llvm-options", "-ppu-dlcm=0"],
    )

    # prefix-key value rules: single operands resolved as a whole
    expect(
        "maxrregcount prefix rule",
        translate_nvcc_options(["-Xptxas", "--maxrregcount=64"]),
        ["--llvm-options", "-ppu-max-vreg-count=64"],
    )
    expect(
        "maxntid single dimension",
        translate_nvcc_options(["-Xptxas", "--maxntid=128"]),
        ["--llvm-options", "-ppu-maxntid=128"],
    )
    expect(
        "maxntid multi-dimension dropped (unrepresentable single-uint target)",
        translate_nvcc_options(["-Xptxas", "--maxntid=128,4"]),
        [],
    )
    expect(
        "minnctapersm prefix rule",
        translate_nvcc_options(["-Xptxas", "--minnctapersm=2"]),
        ["--llvm-options", "-ppu-minblockscu=2"],
    )
    expect(
        "ptxas warning-as-error",
        translate_nvcc_options(["-Xptxas", "--warning-as-error"]),
        ["--llvm-options", "-fatal-warnings"],
    )
    expect(
        "ptxas -w",
        translate_nvcc_options(["-Xptxas", "-w"]),
        ["--llvm-options", "--no-warn"],
    )
    expect(
        "disable-warnings",
        translate_nvcc_options(["-Xptxas", "--disable-warnings"]),
        ["--llvm-options", "--no-warn"],
    )

    # ptxas drop rules consume the value together with the key
    expect(
        "ptxas register-usage-level dropped with value",
        translate_nvcc_options(["-Xptxas", "--register-usage-level=10", "--keep"]),
        ["--keep"],
    )
    expect(
        "ptxas position-independent-code dropped",
        translate_nvcc_options(["-Xptxas", "--position-independent-code", "--keep"]),
        ["--keep"],
    )

    # --Werror: comma lists split per kind, ignored kinds dropped
    expect(
        "Werror single kind",
        translate_nvcc_options(["--Werror", "reorder"]),
        ["--Werror", "reorder"],
    )
    expect(
        "Werror comma list with ignored kind",
        translate_nvcc_options(["--Werror", "reorder,cross-execution-space-call"]),
        ["--Werror", "reorder"],
    )
    expect(
        "Werror ignored kind alone",
        translate_nvcc_options(["--Werror", "cross-execution-space-call"]),
        [],
    )

    # dropped with value: no hgcc --fsanitize/--apply-controls on this SDK
    expect(
        "sanitize dropped",
        translate_nvcc_options(["--sanitize", "memcheck"]),
        [],
    )
    expect(
        "apply-controls dropped",
        translate_nvcc_options(["--apply-controls", "off"]),
        [],
    )

    # -gencode is arch-targeting; callers pass --gpu-architecture instead
    expect(
        "gencode dropped with value",
        translate_nvcc_options(["-gencode", "arch=compute_80,code=sm_80"]),
        [],
    )

    # ignored flags, with and without values, long and short
    expect(
        "ignored with value (short)",
        translate_nvcc_options(["--diag-suppress", "1234", "-Wno-deprecated-declarations"]),
        [],
    )
    expect(
        "ignored joined form",
        translate_nvcc_options(["--diag-suppress=1234"]),
        [],
    )

    # -Xcompiler value forwarded raw (hgcc splits commas itself)
    expect(
        "Xcompiler raw value",
        translate_nvcc_options(["-Xcompiler", "-fPIC,-std=c++17"]),
        ["-Xcompiler", "-fPIC,-std=c++17"],
    )

    # whole option string input (shlex-split)
    expect(
        "string input",
        translate_nvcc_options("-O3 -std=c++17 -Xptxas -v,--register-usage-level=10 -Wno-deprecated-declarations"),
        ["-O3", "-std=c++17", "--llvm-options", "--resource-usage"],
    )

    # repeated -Xptxas occurrences accumulate
    expect(
        "repeated occurrences",
        translate_nvcc_options(["-Xptxas", "-v", "-Xptxas", "-dlcm=cg"]),
        ["--llvm-options", "--resource-usage", "--llvm-options", "-ppu-dlcm=2"],
    )

    # mixed comma lists: dropped keys must not swallow the other elements,
    # and dimension operands stay with their key in both orderings
    expect(
        "comma list maxrregcount then -v",
        translate_nvcc_options(["-Xptxas=--maxrregcount=64,-v"]),
        ["--llvm-options", "-ppu-max-vreg-count=64", "--llvm-options", "--resource-usage"],
    )
    expect(
        "comma list dropped key then -v",
        translate_nvcc_options(["-Xptxas=--register-usage-level=10,-v"]),
        ["--llvm-options", "--resource-usage"],
    )
    expect(
        "comma list -v then multi-dim maxntid",
        translate_nvcc_options(["-Xptxas=-v,--maxntid=128,4"]),
        ["--llvm-options", "--resource-usage"],
    )
    expect(
        "comma list multi-dim maxntid then -v",
        translate_nvcc_options(["-Xptxas=--maxntid=128,4,-v"]),
        ["--llvm-options", "--resource-usage"],
    )

    # operand-taking keys pair with a following bare value, in every form
    # nvcc accepts: separated, comma-joined, and across repeated occurrences
    expect(
        "operand paired after key",
        translate_nvcc_options(["-Xptxas", "--maxrregcount", "64", "-c", "a.cu"]),
        ["-c", "a.cu", "--llvm-options", "-ppu-max-vreg-count=64"],
    )
    expect(
        "operand paired via comma",
        translate_nvcc_options(["-Xptxas=--maxrregcount,64"]),
        ["--llvm-options", "-ppu-max-vreg-count=64"],
    )
    expect(
        "operand paired across occurrences",
        translate_nvcc_options(["-Xptxas", "--maxrregcount", "-Xptxas", "64"]),
        ["--llvm-options", "-ppu-max-vreg-count=64"],
    )
    expect(
        "maxntid separated operand",
        translate_nvcc_options(["-Xptxas", "--maxntid", "128"]),
        ["--llvm-options", "-ppu-maxntid=128"],
    )

    # space-separated option list inside one value, like a quoted -Xptxas arg
    expect(
        "space-separated list value",
        translate_nvcc_options("-Xptxas '-v -w'"),
        ["--llvm-options", "--resource-usage", "--llvm-options", "--no-warn"],
    )

    # --options-file contents are spliced in and translated
    optfile = os.path.join(tmp, "nvcc.opts")
    with open(optfile, "w") as fh:
        fh.write("-Xptxas -v\n-O3 -std=c++14\n")
    expect(
        "options-file contents spliced in",
        translate_nvcc_options(["-optf", optfile, "-c", "a.cu"]),
        ["-O3", "-std=c++14", "-c", "a.cu", "--llvm-options", "--resource-usage"],
    )
    expect(
        "options-file joined spelling",
        translate_nvcc_options(["--options-file=" + optfile, "-c", "a.cu"]),
        ["-O3", "-std=c++14", "-c", "a.cu", "--llvm-options", "--resource-usage"],
    )
    try:
        translate_nvcc_options(["-optf", os.path.join(tmp, "missing.opts")])
        failures.append("options-file missing file: no error raised")
    except FileNotFoundError:
        pass

    # repeated values dedupe across occurrences, long/short spellings alike;
    # a repeated setting keeps its final value (hgcc rejects repeats)
    expect(
        "duplicate -v across spellings deduped",
        translate_nvcc_options(["-Xptxas", "-v", "--ptxas-options=-v", "-Xptxas", "-v"]),
        ["--llvm-options", "--resource-usage"],
    )
    expect(
        "repeated setting keeps final value",
        translate_nvcc_options(["-Xptxas=-dlcm=cg,-dlcm=ca,-dlcm=cg"]),
        ["--llvm-options", "-ppu-dlcm=2"],
    )
    expect(
        "distinct settings survive",
        translate_nvcc_options(["-Xptxas", "-v", "-Xptxas", "-dlcm=cg"]),
        ["--llvm-options", "--resource-usage", "--llvm-options", "-ppu-dlcm=2"],
    )

    # forwarding flags registered with their operand
    expect(
        "Xpreprocessor forwards operand",
        translate_nvcc_options(["-Xpreprocessor", "-DFOO=1", "-c", "a.cu"]),
        ["-Xpreprocess", "-DFOO=1", "-c", "a.cu"],
    )

    # restored exported entries
    expect(
        "device-debug mapping restored",
        translate_nvcc_options(["--device-debug"]),
        ["-G", "-D__CUDACC_DEBUG__"],
    )
    expect(
        "relocatable-link mapping restored",
        translate_nvcc_options(["--relocatable-link"]),
        ["-r"],
    )
    expect(
        "dopt mapping restored",
        translate_nvcc_options(["--dopt", "on"]),
        ["-dopt", "3"],
    )

    # an exact value_map key containing a comma wins over list splitting
    import sailify.nvcc_hgcc_options as optmod
    optmod.NVCC_HGCC_OPTIONS["--ptxas-options"]["value_map"]["-v,-w"] = "--resource-usage"
    optmod._OPTION_LOOKUP = None
    try:
        expect(
            "exact whole-value beats splitting",
            translate_nvcc_options(["-Xptxas=-v,-w"]),
            ["--llvm-options", "--resource-usage"],
        )
    finally:
        del optmod.NVCC_HGCC_OPTIONS["--ptxas-options"]["value_map"]["-v,-w"]
        optmod._OPTION_LOOKUP = None

    # direct mappings
    expect(
        "direct mappings",
        translate_nvcc_options(["--cubin", "--expt-extended-lambda"]),
        ["-hgbin", "--extended-lambda", "-D__CUDACC_EXTENDED_LAMBDA__"],
    )
    expect(
        "allow-unsupported-compiler",
        translate_nvcc_options(["--allow-unsupported-compiler"]),
        ["-Xpreprocess", "-D__NV_NO_HOST_COMPILER_CHECK=1"],
    )
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
            ("jit_rerun_inject_recursion", lambda: test_jit_rerun_inject_recursion(tmp)),
            ("sglang_aot_preprocessor_direct", lambda: test_sglang_aot_preprocessor_direct(tmp)),
            ("sdk22_mapping_state_toggles_with_version", test_sdk22_mapping_state_toggles_with_version),
            ("sdk23_renames_toggles_with_version", test_sdk23_renames_toggles_with_version),
            ("sdk222_device_function_mapping", lambda: test_sdk222_device_function_mapping(tmp)),
            ("sdk222_lexical_boundaries", lambda: test_sdk222_lexical_boundaries(tmp)),
            ("sdk222_concurrent_conversion", lambda: test_sdk222_concurrent_conversion(tmp)),
            ("sdk222_detected_compiler_switch", lambda: test_sdk222_detected_compiler_switch(tmp)),
            ("sdk222_explicit_entry_version", lambda: test_sdk222_explicit_entry_version(tmp)),
            ("sdk222_compiler_identity", lambda: test_sdk222_compiler_identity(tmp)),
            ("sdk23_compat_shims", lambda: test_sdk23_compat_shims(tmp)),
            ("sdk22_state_resolved_with_compat_disabled", lambda: test_sdk22_state_resolved_with_compat_disabled(tmp)),
            ("versioned_runtime_api_conversion", lambda: test_versioned_runtime_api_conversion(tmp)),
            ("nvcc_flag_mapping", lambda: test_nvcc_flag_mapping(tmp)),
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

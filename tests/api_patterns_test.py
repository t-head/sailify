import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

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
    from sailify.cuda_to_ppu_mappings import PPU_SDK_VERSION_MAPPINGS

    for kind, _, mapping in PPU_SDK_VERSION_MAPPINGS:
        if kind == "identifier":
            for src in mapping:
                untouched.append(src + ";")
                if rx.match(src + ";") is None or _PPU_MAP.get(src) != src:
                    failures.append(src + " missing identity entry in codegen map/trie")
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


def test_sdk_version_mapping_registry():
    import sailify.cuda_to_ppu_mappings as mappings
    import sailify.sailify_python as sp

    failures = []
    groups = (
        ("identifier", (2, 2, 0), mappings.PPU_SDK_2_2_IDENTIFIER_MAP),
        ("identifier", (2, 2, 2), mappings.PPU_SDK_2_2_2_IDENTIFIER_MAP),
        ("include", (2, 3, 0), mappings.PPU_SDK_2_3_RENAME_MAP),
    )
    if mappings.PPU_SDK_VERSION_MAPPINGS != groups:
        failures.append("SDK registry must contain the three versioned mapping groups")
    snapshots = [dict(mapping) for _, _, mapping in groups]
    with patch.object(sp, "detect_ppu_sdk_version", return_value=(2, 3, 0)) as detect:
        for version in _SDK_VERSIONS + _SDK_SWITCH_VERSIONS:
            for kind in ("identifier", "include"):
                expected = {}
                for group_kind, minimum, mapping in groups:
                    if kind == group_kind and version is not None and version >= minimum:
                        expected.update(mapping)
                got = sp._get_versioned_mappings(kind, version)
                if got != expected:
                    failures.append("registry %s version=%r: %r != %r" %
                                    (kind, version, got, expected))
        if detect.called:
            failures.append("mapping resolver called version detector")
    if snapshots != [dict(mapping) for _, _, mapping in groups]:
        failures.append("mapping resolution mutated registry data")
    return failures


_SDK222_ERF_MAP = {
    "__nv_" + name: "__hg_" + name
    for name in ("erf", "erff", "erfc", "erfcf", "erfinv", "erfinvf",
                 "erfcinv", "erfcinvf", "erfcx", "erfcxf")
}
_SDK_VERSIONS = (None, (2, 1, 5), (2, 2, 0), (2, 2, 1),
                 (2, 2, 2), (2, 2, 3), (2, 3, 0))
_SDK_SWITCH_VERSIONS = ((2, 3, 0), (2, 1, 5), None, (2, 3, 0))
_SDK_VERSIONED_SOURCE = (
    '#include <vector_types.h>\n#include_next "sub/vector_types.h"\n'
    '__NV_THREAD_SCOPE_SYSTEM; __nv_erff(x); __nv_atomic_add(p, 1, 0, 0);\n'
)


def _sdk_version_expected(version):
    expected = _SDK_VERSIONED_SOURCE.replace("__nv_atomic_add", "__hg_atomic_add")
    if version is not None and version >= (2, 2, 0):
        expected = expected.replace("__NV_THREAD_SCOPE_SYSTEM", "__HG_THREAD_SCOPE_SYSTEM")
    if version is not None and version >= (2, 2, 2):
        expected = expected.replace("__nv_erff", "__hg_erff")
    if version is not None and version >= (2, 3, 0):
        expected = expected.replace("vector_types.h", "hggc_vector_types.h")
    return expected


def _sdk222_convert(sp, tmp, source, **version):
    path = os.path.join(tmp, "sdk222_direct.cu")
    with open(path, "w") as fh:
        fh.write(source)
    result = sp.preprocessor(
        output_directory=tmp, filepath=path, all_files=[], stats={},
        clean_ctx=None, show_progress=False, project_directory=tmp, **version,
    )
    if result.status != "ok":
        raise RuntimeError("SDK 2.2.2 preprocessor status: " + str(result.status))
    with open(result.ppuified_path) as fh:
        return fh.read()


def test_sdk222_erf_direct(tmp):
    from sailify.cuda_to_ppu_mappings import PPU_SDK_2_2_2_IDENTIFIER_MAP
    import sailify.sailify_python as sp

    failures = []
    if dict(PPU_SDK_2_2_2_IDENTIFIER_MAP) != _SDK222_ERF_MAP:
        failures.append("SDK 2.2.2 erf mapping must contain exactly the ten names")
    calls = " ".join(name + "(x);" for name in _SDK222_ERF_MAP) + "\n"
    untouched = " ".join("prefix" + n + " " + n + "_suffix " + n + "2"
                         for n in (*_SDK222_ERF_MAP, "__NV_THREAD_SCOPE_SYSTEM")) + "\n"
    untouched += "__nv_sqrtf(x); __nv_fmaf(x, y, z); "
    untouched += " ".join(n[5:] + "(x);" for n in _SDK222_ERF_MAP) + "\n"
    source = _SDK_VERSIONED_SOURCE + calls + untouched
    options_list = [{"ppu_sdk_version": v} for v in _SDK_VERSIONS + _SDK_SWITCH_VERSIONS]
    # Omission after an enabled conversion must also leave every conditional group off.
    for options in options_list + [{}]:
        version = options.get("ppu_sdk_version")
        enabled = version is not None and version >= (2, 2, 2)
        expected = _sdk_version_expected(version)
        expected += (calls.replace("__nv_", "__hg_") if enabled else calls) + untouched
        with patch.object(sp, "detect_ppu_sdk_version", return_value=(2, 3, 0)) as detect:
            got = _sdk222_convert(sp, tmp, source, **options)
            again = _sdk222_convert(sp, tmp, got, **options)
        label = "options=%r" % options
        if got != expected or again != got:
            failures.append(label + ": versioned names/includes/boundaries/idempotence mismatch")
        if detect.called:
            failures.append(label + ": direct preprocessor called version detector")
    return failures


def test_conditional_identifier_include_mask(tmp):
    from sailify.cuda_to_ppu_mappings import PPU_SDK_2_2_IDENTIFIER_MAP
    import sailify.sailify_python as sp

    protected = (
        '#include "__nv_erf.h"\n#include <__nv_erff.h>\n'
        '#include_next "__nv_erfc.h"\n#include_next <__nv_erfcf.h>\n'
        ' \t# include "__nv_erfinv.h"\n'
        '# include_next <__nv_erfinvf.h>\n'
        '#include"detail/__nv_erfcinv.h"\n'
        '#include_next<detail/__nv_erfcinvf.h>\n'
        '#include "detail//__nv_erfcx.h"\n'
        '#include <detail/__nv_erfcxf.h>\n'
        "'__nv_erff'; '\\''; // __nv_erfc(x)\n/* __nv_erfinv(x) */\n"
    )
    mapped = (
        'const char* normal = "__nv_erff(x)";\n'
        'const char* raw = R"tag(\n#include "__nv_erfcf.h"\n__nv_erfcf(x))tag";\n'
        '#define APPLY(x) \\\n (__nv_erfinvf(x) + cudaMalloc(&p, n))\n'
        '__nv_erf(x);\n'
    )
    for name in PPU_SDK_2_2_IDENTIFIER_MAP:
        protected += '#include "%s.h"\n# include_next<detail/%s.h>\n' % (name, name)
    protected += "'__NV_THREAD_SCOPE_THREAD'; // __NV_THREAD_SCOPE_BLOCK\n"
    protected += "/* __NV_THREAD_SCOPE_CLUSTER */\n"
    mapped += 'const char* scope = "__NV_THREAD_SCOPE_SYSTEM";\n'
    mapped += '#define SCOPE __NV_THREAD_SCOPE_DEVICE\n'
    failures = []
    for version in _SDK_VERSIONS:
        expected = mapped.replace("cudaMalloc", "hggcMalloc")
        if version is not None and version >= (2, 2, 0):
            expected = expected.replace("__NV_THREAD_SCOPE_", "__HG_THREAD_SCOPE_")
        if version is not None and version >= (2, 2, 2):
            expected = expected.replace("__nv_", "__hg_")
        got = _sdk222_convert(sp, tmp, protected + mapped, ppu_sdk_version=version)
        if got != protected + expected:
            failures.append("SDK %r: conditional include/comment/char shielding or string/macro mapping mismatch" %
                            (version,))
    return failures


def test_sdk222_erf_entry_points(tmp):
    import sailify.sailify_python as sp

    failures = []
    calls = " ".join(n + "(x);" for n in _SDK222_ERF_MAP) + "\n"
    body = _SDK_VERSIONED_SOURCE + calls
    for route in ("project", "extra_files", "recursive", "explicit_project"):
        for idx, version in enumerate(_SDK_VERSIONS):
            root = os.path.join(tmp, "sdk222_%s_%d" % (route, idx))
            os.makedirs(root, exist_ok=True)
            kernel, header = [os.path.join(root, n) for n in ("kernel.cu", "detail.cuh")]
            sources = {kernel: '#include "detail.cuh"\n' + body, header: body}
            for path, text in sources.items():
                with open(path, "w") as fh:
                    fh.write(text)
            enabled = version is not None and version >= (2, 2, 2)
            explicit = route == "explicit_project" and version is not None
            detected = ((2, 1, 5) if enabled else (2, 3, 0)) if explicit else version
            with patch.object(sp, "detect_ppu_sdk_version", return_value=detected) as detect:
                if route in ("project", "explicit_project"):
                    options = {"ppu_sdk_version": version} if route == "explicit_project" else {}
                    results = sp.sailify(root, install_ppu_compat=False, **options)
                elif route == "extra_files":
                    results = sp.sailify_extra_files(root, [kernel, header], install_ppu_compat=False)
                else:
                    results = sp.sailify_extra_files_recursive(root, [kernel], install_ppu_compat=False)
            if detect.called == explicit:
                failures.append("%s version=%r: detection did not respect explicit/default version" %
                                (route, version))
            expected_body = _sdk_version_expected(version)
            expected_body += calls.replace("__nv_", "__hg_") if enabled else calls
            for path, text in sources.items():
                label = "%s version=%r %s" % (route, version, os.path.basename(path))
                if path not in results or results[path].status != "ok":
                    failures.append(label + ": missing/failed conversion (including header closure)")
                    continue
                with open(results[path].ppuified_path) as fh:
                    got = fh.read()
                expected = text.replace(body, expected_body)
                if got != expected:
                    failures.append(label + ": selected SDK version not respected")
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
    ("#include_next <vector_types.h>", "#include_next <hggc_vector_types.h>"),
    ('#include_next "cuda_runtime.h"', '#include_next "hggc_runtime.h"'),
]


def test_sdk23_include_version_matrix():
    import sailify.sailify_python as sp

    failures = []
    versions = _SDK_VERSIONS + ((2, 2, 9), (2, 3, 1)) + _SDK_SWITCH_VERSIONS
    with patch.object(sp, "detect_ppu_sdk_version", return_value=(2, 3, 0)) as detect:
        for version in versions:
            for src, renamed in _SDK23_RENAME_CASES:
                expected = (renamed if version is not None and version >= (2, 3, 0)
                            else src.replace("cuda_runtime.h", "hggc_runtime.h"))
                got = sp._rewrite_include_directives(src, ppu_sdk_version=version)
                if got != expected:
                    failures.append("SDK %r include %r -> %r, expected %r" %
                                    (version, src, got, expected))
        for src, _ in _SDK23_RENAME_CASES:
            if sp._rewrite_include_directives(src) != src.replace("cuda_runtime.h", "hggc_runtime.h"):
                failures.append("omitted SDK version activated conditional include rename: " + src)
        if detect.called:
            failures.append("include rewrite helper called version detector")
    return failures


def test_include_next_conversion(tmp):
    import sailify.sailify_python as sp

    source = (
        '#include_next <cuda_runtime.h>\n'
        '# include_next "cuda_runtime.h"\n'
        '#include_next<vector_types.h>\n'
        '# include_next "sub/vector_types.h"\n'
        '#include_next "__nv_erff.h"\n'
    )
    failures = []
    for version in _SDK_VERSIONS + _SDK_SWITCH_VERSIONS:
        expected = source.replace("cuda_runtime.h", "hggc_runtime.h")
        if version is not None and version >= (2, 3, 0):
            expected = expected.replace("vector_types.h", "hggc_vector_types.h")
        rewritten = sp._rewrite_include_directives(source, ppu_sdk_version=version)
        converted = _sdk222_convert(sp, tmp, source, ppu_sdk_version=version)
        if rewritten != expected or converted != expected:
            failures.append("include_next SDK %r: rewrite=%r converted=%r" %
                            (version, rewritten, converted))
    return failures


def test_include_comment_splice_operands(tmp):
    from sailify.cuda_to_ppu_mappings import PPU_SDK_2_2_IDENTIFIER_MAP
    import sailify.sailify_python as sp

    # Keep the operand separate so only a *mapped* header may lose its splices.
    cases = []
    for directive in ("include", "include_next"):
        for opening, closing in (('"', '"'), ('<', '>')):
            prefix = ' /* lead */ #/* hash */' + directive + '/* operand */' + opening
            suffix = closing + ' /* trailing __nv_erff */\n'
            for header in ("__nv_erff.h", "detail/vector_types.h", "cuda_runtime.h"):
                cases.append((prefix, header, suffix))
            for name in PPU_SDK_2_2_IDENTIFIER_MAP:
                cases.append((prefix, "detail/" + name + ".h", suffix))
            for newline in ("\n", "\r\n"):
                splice = "\\" + newline
                prefix = ('\t#' + splice + ' /* hash */inc' + splice +
                          directive[3:] + ' /* before' + newline +
                          ' operand */' + splice + opening)
                suffix = closing + ' /* after */' + splice + newline
                for header in ("detail/__nv_erff." + splice + "h",
                               "detail/__nv_" + splice + "erff.h",
                               "__NV_THREAD_SCOPE_SYSTEM." + splice + "h",
                               "sub/vector_" + splice + "types.h",
                               "cuda_run" + splice + "time.h"):
                    cases.append((prefix, header, suffix))
                    probe = ('#if __has_inc' + splice + directive[3:] +
                             ' /* name */ (' + splice + ' /* operand */' + opening)
                    cases.append((probe, header, closing + ' /* close */)' + newline +
                                  '#endif' + newline))

    failures = []
    for version in (None, (2, 2, 1), (2, 2, 2), (2, 3, 0)):
        separated = ('/* first */\n#include "__nv_erff.h"\n#include <cuda_runtime.h>\n'
                     '/* second */ #include_next <cuda_runtime.h>\n')
        expected = separated.replace("cuda_runtime.h", "hggc_runtime.h")
        for got in (sp._rewrite_include_directives(separated, ppu_sdk_version=version),
                    _sdk222_convert(sp, tmp, separated, ppu_sdk_version=version)):
            if got != expected:
                failures.append("SDK %r: block comment swallowed a directive: %r" % (version, got))
        for prefix, header, suffix in cases:
            logical = re.sub(r"\\\r?\n", "", header)
            expected_header = logical.replace("cuda_runtime.h", "hggc_runtime.h")
            if version is not None and version >= (2, 3, 0):
                expected_header = expected_header.replace("vector_types.h", "hggc_vector_types.h")
            mapped = expected_header != logical
            if not mapped:
                expected_header = header
            source = prefix + header + suffix
            for route in ("rewrite", "preprocessor"):
                if route == "rewrite":
                    convert = lambda text: sp._rewrite_include_directives(text, ppu_sdk_version=version)
                    before, after, operand = prefix, suffix, expected_header
                else:
                    convert = lambda text: _sdk222_convert(sp, tmp, text, ppu_sdk_version=version)
                    # Existing file I/O normalizes CRLF; the helper above checks exact bytes.
                    before, after, operand = (s.replace("\r\n", "\n")
                                               for s in (prefix, suffix, expected_header))
                got = convert(source)
                label = "%s SDK %r source=%r" % (route, version, source)
                if not got.startswith(before) or not got.endswith(after):
                    failures.append(label + ": changed directive/comments/delimiters: " + repr(got))
                else:
                    actual_header = got[len(before):len(got) - len(after)]
                    if mapped:
                        actual_header = re.sub(r"\\\r?\n", "", actual_header)
                    if actual_header != operand:
                        failures.append(label + ": header %r != %r" % (actual_header, operand))
                if convert(got) != got:
                    failures.append(label + ": not idempotent")
    return failures


def test_has_include_literal_operands(tmp):
    from sailify.cuda_to_ppu_mappings import PPU_SDK_2_2_IDENTIFIER_MAP
    import sailify.sailify_python as sp

    headers = (
        '#if __has_include("vector_types.h") && __has_include_next(<sub/vector_types.h>)\n'
        '#include "vector_types.h"\n#include_next <sub/vector_types.h>\n'
        '#elif __has_include(<cuda_runtime.h>) || __has_include_next("cuda_runtime.h")\n'
        '#include <cuda_runtime.h>\n#include_next "cuda_runtime.h"\n#endif\n'
        '#define HAVE_VECTOR __has_include /* name */ ( /* open */ "sub/vector_types.h" /* close */ )\n'
        '#define HAVE_RUNTIME __has_include_next/* name */(/* open */<cuda_runtime.h>/* close */)\n'
    )
    protected = (
        '#define HEADER_NAME <vector_types.h>\n'
        '#if __has_include(HEADER_NAME) || __has_include_next(HEADER_NAME)\n'
        '#include HEADER_NAME\n#endif\n'
        '#if __has_include /* first */ (HEADER_NAME) // __has_include /* second */ (<cuda_runtime.h>)\n'
        '#endif\n'
        '#if my__has_include("vector_types.h") || __has_include_extra(<vector_types.h>)\n'
        '#elif my__has_include_next(<vector_types.h>) || __has_include_next_extra("vector_types.h")\n'
        '#endif\n'
        '// __has_include(<cuda_runtime.h>) __has_include_next("vector_types.h")\n'
        '/*\n#if __has_include("vector_types.h") || __has_include_next(<cuda_runtime.h>)\n#endif\n*/\n'
        "'__has_include(<cuda_runtime.h>)'; '__has_include_next(<__nv_erff.h>)';\n"
    )
    for name in ("__nv_erff", *PPU_SDK_2_2_IDENTIFIER_MAP):
        protected += (
            '#if __has_include("%s.h") || __has_include_next(<detail/%s.h>)\n'
            '#elif __has_include(<%s.h>) || __has_include_next("detail/%s.h")\n'
            '#endif\n#define HAVE_USER_%s __has_include("%s.h")\n'
        ) % (name, name, name, name, name, name)
    for newline in ("\n", "\r\n"):
        splice = "\\" + newline
        headers += ('#if __has_in' + splice + 'clude /* name */ (' + splice +
                    ' <vector_types.h> /* close */) || __has_include_' + splice +
                    'next/* name */("cuda_runtime.h")' + newline + '#endif' + newline)
        protected += ('#define HAVE_SPLICED __has_in' + splice + 'clude_next /* name */ (' +
                      splice + ' /* open */ <detail/__nv_erff.' + splice + 'h>)' + newline)
    mapped = (
        '__nv_erff(x); __NV_THREAD_SCOPE_SYSTEM; cudaMalloc(&p, n);\n'
        'my__has_include("__nv_erff.h"); __has_include_extra(<__nv_erff.h>);\n'
        'my__has_include_next("__nv_erff.h"); __has_include_next_extra(<__nv_erff.h>);\n'
        'const char* normal = "__has_include(<__nv_erff.h>) __nv_erff(x) cudaMalloc(&p, n)";\n'
        'const char* include_text = "#include <__nv_erff.h> __NV_THREAD_SCOPE_SYSTEM";\n'
        'const char* raw = R"tag(\n#include "__nv_erff.h"\n'
        '#include_next <__NV_THREAD_SCOPE_SYSTEM.h>\n'
        '#if __has_include("__nv_erff.h") || __has_include_next(<__NV_THREAD_SCOPE_SYSTEM.h>)\n'
        '__nv_erff(x); cudaMalloc(&p, n);\n#endif\n)tag";\n'
    )
    strings = (
        'const char* probe_text = "__has_include(<vector_types.h>)";\n'
        'const char* raw_headers = R"tag(\n#if __has_include_next(<vector_types.h>)\n'
        '#include_next <vector_types.h>\n#endif\n)tag";\n'
    )
    source = headers + protected + mapped + strings
    failures = []

    def check(label, got, expected):
        if got != expected:
            offset = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
                          min(len(got), len(expected)))
            failures.append("%s at %d: %r != %r" %
                            (label, offset, got[offset:offset + 100], expected[offset:offset + 100]))

    for version in (None, (2, 2, 1), (2, 2, 2), (2, 3, 0)):
        expected_headers = headers.replace("cuda_runtime.h", "hggc_runtime.h")
        expected_strings = strings
        if version is not None and version >= (2, 3, 0):
            expected_headers = expected_headers.replace("vector_types.h", "hggc_vector_types.h")
            expected_strings = strings.replace("vector_types.h", "hggc_vector_types.h")
        rewritten = sp._rewrite_include_directives(source, ppu_sdk_version=version)
        expected = expected_headers + protected + mapped + expected_strings
        check("SDK %r: literal probe/include rewrite" % (version,), rewritten, expected)
        if sp._rewrite_include_directives(rewritten, ppu_sdk_version=version) != rewritten:
            failures.append("SDK %r: probe rewrite not idempotent" % (version,))
        expected_mapped = mapped.replace("cudaMalloc", "hggcMalloc")
        if version is not None and version >= (2, 2, 0):
            expected_mapped = expected_mapped.replace("__NV_THREAD_SCOPE_", "__HG_THREAD_SCOPE_")
        if version is not None and version >= (2, 2, 2):
            expected_mapped = expected_mapped.replace("__nv_erff", "__hg_erff")
        expected = (expected_headers + protected + expected_mapped + expected_strings).replace("\r\n", "\n")
        got = _sdk222_convert(sp, tmp, source, ppu_sdk_version=version)
        check("SDK %r: probe shielding/API/string/raw conversion" % (version,), got, expected)
        if _sdk222_convert(sp, tmp, got, ppu_sdk_version=version) != got:
            failures.append("SDK %r: probe conversion not idempotent" % (version,))
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


def test_sdk_versions_with_compat_disabled(tmp):
    import sailify.sailify_python as sp

    src_dir = os.path.join(tmp, "compat_off")
    os.makedirs(src_dir, exist_ok=True)
    src = os.path.join(src_dir, "k.cu")
    failures = []
    for version in _SDK_SWITCH_VERSIONS:
        with open(src, "w") as fh:
            fh.write(_SDK_VERSIONED_SOURCE)
        with patch.object(sp, "detect_ppu_sdk_version", return_value=version):
            results = sp.sailify_extra_files(src_dir, [src], install_ppu_compat=False)
        with open(results[src].ppuified_path) as fh:
            text = fh.read()
        if results[src].status != "ok" or text != _sdk_version_expected(version):
            failures.append("compat-disabled conversion ignored SDK %r" % (version,))
        with patch.object(sp, "detect_ppu_sdk_version", return_value=(2, 3, 0)) as detect:
            direct = _sdk222_convert(sp, tmp, _SDK_VERSIONED_SOURCE)
        if direct != _sdk_version_expected(None) or detect.called:
            failures.append("direct conversion inherited high-level SDK %r or detected a version" %
                            (version,))
    if os.path.exists(os.path.join(src_dir, ".ppu_compat")):
        failures.append("install_ppu_compat=False still generated compatibility headers")
    return failures


def test_sdk_direct_concurrent_versions(tmp):
    import sailify.sailify_python as sp

    # Exercise per-call SDK isolation only; other project-wide state is out of scope.
    versions = ((2, 3, 0), (2, 2, 2), (2, 1, 5))
    roots = [os.path.join(tmp, "sdk_concurrent_%d" % i) for i in range(len(versions))]
    for root in roots:
        os.makedirs(root, exist_ok=True)
    barrier = Barrier(len(versions), timeout=10)
    preprocessor = sp.preprocessor

    def synchronized_preprocessor(*args, **kwargs):
        barrier.wait()
        return preprocessor(*args, **kwargs)

    def convert(item):
        root, version = item
        return [_sdk222_convert(sp, root, _SDK_VERSIONED_SOURCE, ppu_sdk_version=version)
                for _ in range(3)]

    with patch.object(sp, "preprocessor", new=synchronized_preprocessor), \
            patch.object(sp, "detect_ppu_sdk_version", side_effect=AssertionError("direct detection")):
        with ThreadPoolExecutor(max_workers=len(versions)) as pool:
            outputs = list(pool.map(convert, zip(roots, versions)))
    return ["concurrent direct conversion leaked SDK selection for %r" % (version,)
            for version, texts in zip(versions, outputs)
            if texts != [_sdk_version_expected(version)] * 3]


def test_sdk_future_registry_groups(tmp):
    # Mock only registry data before sailify_python builds its map/trie at import.
    code = r'''
import os
import re
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import sailify.cuda_to_ppu_mappings as mappings
mappings.PPU_SDK_VERSION_MAPPINGS += (
    ("identifier", (9, 0, 0), {"__nv_future_probe": "__hg_future_probe"}),
    ("include", (9, 0, 0), {"future_probe.h": "hg_future_probe.h"}),
)
import sailify.sailify_python as sp
rx = re.compile(rf"({sp._PPU_TRIE.export_to_regex()})(?=\W)")
assert rx.match("__nv_future_probe();")
assert rx.sub(lambda m: sp._PPU_MAP[m.group(0)], "__nv_future_probe();") == "__nv_future_probe();"
source = ('#include <future_probe.h>\n#include_next "sub/future_probe.h"\n'
          '#include "__nv_future_probe.h"\n#include <vector_types.h>\n'
          '__nv_future_probe(); __NV_THREAD_SCOPE_SYSTEM; __nv_erff(x);\n')
path = os.path.join(sys.argv[2], "future_registry.cu")
with patch.object(sp, "detect_ppu_sdk_version", side_effect=AssertionError("direct detection")):
    for version in ((9, 0, 0), (2, 3, 0), None, (9, 0, 0)):
        expected = source
        if version is not None:
            expected = expected.replace("vector_types.h", "hggc_vector_types.h")
            expected = expected.replace("__NV_THREAD_SCOPE_SYSTEM", "__HG_THREAD_SCOPE_SYSTEM")
            expected = expected.replace("__nv_erff", "__hg_erff")
        if version == (9, 0, 0):
            expected = expected.replace("<future_probe.h>", "<hg_future_probe.h>")
            expected = expected.replace("sub/future_probe.h", "sub/hg_future_probe.h")
            expected = expected.replace("__nv_future_probe()", "__hg_future_probe()")
        with open(path, "w") as fh:
            fh.write(source)
        result = sp.preprocessor(
            output_directory=sys.argv[2], filepath=path, all_files=[], stats={},
            clean_ctx=None, show_progress=False, project_directory=sys.argv[2],
            ppu_sdk_version=version,
        )
        assert result.status == "ok", result.status
        with open(result.ppuified_path) as fh:
            got = fh.read()
        assert got == expected, (version, got, expected)
'''
    result = subprocess.run(
        [sys.executable, "-c", code, os.path.dirname(HERE), tmp],
        capture_output=True, text=True, timeout=30,
    )
    return [] if result.returncode == 0 else ["future registry groups: " + result.stdout + result.stderr]


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
            ("sdk_version_mapping_registry", test_sdk_version_mapping_registry),
            ("sdk222_erf_direct", lambda: test_sdk222_erf_direct(tmp)),
            ("conditional_identifier_include_mask", lambda: test_conditional_identifier_include_mask(tmp)),
            ("sdk222_erf_entry_points", lambda: test_sdk222_erf_entry_points(tmp)),
            ("sdk23_include_version_matrix", test_sdk23_include_version_matrix),
            ("include_next_conversion", lambda: test_include_next_conversion(tmp)),
            ("include_comment_splice_operands", lambda: test_include_comment_splice_operands(tmp)),
            ("has_include_literal_operands", lambda: test_has_include_literal_operands(tmp)),
            ("sdk23_compat_shims", lambda: test_sdk23_compat_shims(tmp)),
            ("sdk_versions_with_compat_disabled", lambda: test_sdk_versions_with_compat_disabled(tmp)),
            ("sdk_direct_concurrent_versions", lambda: test_sdk_direct_concurrent_versions(tmp)),
            ("sdk_future_registry_groups", lambda: test_sdk_future_registry_groups(tmp)),
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

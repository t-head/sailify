"""Tests for the unsupported-API stub probe (sailify.unsupported_stubs.probe)."""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.abspath(REPO))

from sailify.unsupported_stubs import generate_stubs as G
from sailify.unsupported_stubs import probe as P


def _report(npass, nfail):
    print("PROBE_SUMMARY pass=" + str(npass) + " fail=" + str(nfail))
    return 0 if nfail == 0 else 1


class _Env:
    """Temporarily isolate PATH and the SDK environment variables.

    All controlled variables are cleared first, then only the specified
    ones are set — inherited values (e.g. from a sourced envsetup.sh) must
    not leak into the test.
    """

    KEYS = ("PATH", "PPU_PATH", "PPU_SDK", "PPU_HOME")

    def __init__(self, **values):
        self._values = values
        self._saved = {}

    def __enter__(self):
        for k in self.KEYS:
            if k in os.environ:
                self._saved[k] = os.environ[k]
            os.environ.pop(k, None)
        for k, v in self._values.items():
            if v is not None:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k in self.KEYS:
            os.environ.pop(k, None)
            if k in self._saved:
                os.environ[k] = self._saved[k]


_sdk_counter = [0]


def _make_sdk(root, version="2.2", with_acblas=True):
    """A fake SDK layout: bin/hgcc printing 'Release version <version>' plus
    host libraries compiled as real shared objects exporting cublas/cudnn names."""
    _sdk_counter[0] += 1
    sdk = os.path.join(root, "sdk_%d_%s" % (_sdk_counter[0], version.replace(".", "_")))
    os.makedirs(os.path.join(sdk, "bin"))
    os.makedirs(os.path.join(sdk, "lib"))
    with open(os.path.join(sdk, "bin", "hgcc"), "w") as f:
        f.write("#!/bin/sh\necho 'Release version %s'\n" % version)
    os.chmod(os.path.join(sdk, "bin", "hgcc"), 0o755)
    if with_acblas:
        src = os.path.join(sdk, "host.c")
        with open(src, "w") as f:
            f.write("int acblasSetKernelStream(int s) { return s; }\n"
                    "int acdnnBuildRNNDynamic(int x) { return x; }\n")
        for so in ("libacblas.so", "libacdnn.so", "libhggc.so"):
            r = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o",
                 os.path.join(sdk, "lib", so), src],
                capture_output=True, check=False,
            )
            if r.returncode != 0:
                raise RuntimeError("gcc failed: " + r.stderr.decode())
    return sdk


_wrap_counter = [0]


def _make_wrapper(root, target):
    _wrap_counter[0] += 1
    d = os.path.join(root, "wrapbin_%d" % _wrap_counter[0])
    os.makedirs(d)
    # A regular file wrapper — realpath is a no-op, so ../lib is unrelated.
    with open(os.path.join(d, "hgcc"), "w") as f:
        f.write("#!/bin/sh\nexec '%s/bin/hgcc' \"$@\"\n" % target)
    os.chmod(os.path.join(d, "hgcc"), 0o755)
    # The system-looking ../lib next to the wrapper, without PPU libraries.
    os.makedirs(os.path.join(root, "lib"), exist_ok=True)
    return d


def test_sdk_dir_via_wrapper_and_env(tmp):
    failures = []
    sdk = _make_sdk(tmp)
    wrap = _make_wrapper(tmp, sdk)
    with _Env(PATH=wrap + ":/usr/bin:/bin", PPU_PATH=sdk):
        d = P.find_sdk_lib_dir()
        if d != os.path.join(sdk, "lib"):
            failures.append("wrapper hgcc + matching PPU_PATH: got %r" % (d,))
    with _Env(PATH="/usr/bin:/bin", PPU_PATH=sdk):
        d = P.find_sdk_lib_dir()
        if d != os.path.join(sdk, "lib"):
            failures.append("no hgcc + PPU_PATH: got %r" % (d,))
    return failures


def test_sdk_dir_via_symlinked_hgcc(tmp):
    failures = []
    sdk = _make_sdk(tmp)
    d = os.path.join(tmp, "symbin")
    os.makedirs(d)
    os.symlink(os.path.join(sdk, "bin", "hgcc"), os.path.join(d, "hgcc"))
    with _Env(PATH=d + ":/usr/bin:/bin", PPU_PATH=None):
        got = P.find_sdk_lib_dir()
        if got != os.path.join(sdk, "lib"):
            failures.append("symlinked hgcc: got %r" % (got,))
    return failures


def test_sdk_dir_rejects_non_sdk_dirs(tmp):
    failures = []
    empty = os.path.join(tmp, "empty_sdk")
    os.makedirs(os.path.join(empty, "lib"))
    with _Env(PATH="/usr/bin:/bin", PPU_PATH=empty):
        d = P.find_sdk_lib_dir()
        if d is not None:
            failures.append("dir without host libs accepted: %r" % (d,))
    return failures


def test_sdk_dir_rejects_version_mismatch(tmp):
    failures = []
    sdk_a = _make_sdk(tmp, version="2.2")
    sdk_b = _make_sdk(tmp, version="2.1")
    wrap = _make_wrapper(tmp, sdk_a)
    with _Env(PATH=wrap + ":/usr/bin:/bin", PPU_PATH=sdk_b):
        d = P.find_sdk_lib_dir()
        if d is not None:
            failures.append("mismatched env SDK accepted: %r" % (d,))
    return failures


def test_resolve_host_per_candidate(tmp):
    failures = []
    sdk = _make_sdk(tmp)
    # libhggcrt1.so is absent but libhggc.so exists -> the latter is probed.
    hosts = P._resolve_host(os.path.join(sdk, "lib"),
                            ["libhggcrt1.so", "libhggc.so"])
    if hosts != [os.path.join(sdk, "lib", "libhggc.so")]:
        failures.append("per-candidate resolve failed: %r" % (hosts,))
    return failures


def test_symbols_strip_version_suffix(tmp):
    failures = []
    libc = None
    for cand in ("/usr/lib64/libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6",
                 "/lib/libc.so.6"):
        if os.path.isfile(cand):
            libc = cand
            break
    if libc is None:
        return []
    syms = P._symbols_of_so(libc)
    if syms is None or "printf" not in syms:
        failures.append("printf missing from libc symbols")
    if any("@" in s for s in (syms or [])):
        failures.append("@version suffix not stripped")
    return failures


def test_probe_supported_apis(tmp):
    failures = []
    sdk = _make_sdk(tmp)
    wrap = _make_wrapper(tmp, sdk)
    apis = [
        {"name": "acblasSetKernelStream", "library": "cublas"},
        {"name": "acblasAsumEx_64", "library": "cublas"},
        {"name": "acdnnBuildRNNDynamic", "library": "cudnn"},
        {"name": "acfftFoo", "library": "cufftw"},
        {"name": "hggcFoo", "library": "cuda"},
    ]
    with _Env(PATH=wrap + ":/usr/bin:/bin", PPU_PATH=sdk):
        r = P.probe_supported_apis(apis)
        if "acblasSetKernelStream" not in r.supported or "cublas" not in r.probed:
            failures.append("exported cublas API not detected: %r" % (sorted(r.supported),))
        if "acblasAsumEx_64" in r.supported:
            failures.append("unexported cublas API detected")
        if "acdnnBuildRNNDynamic" not in r.supported:
            failures.append("exported cudnn API not detected")
        if "cufftw" not in r.unknown:
            failures.append("cufftw (no host lib) not unknown")
        if "cuda" not in r.probed:
            failures.append("cuda not probed via libhggc.so alone")
        hdr = G.generate_stubs_header(stub_probe=True)
        if "acblasSetKernelStream" in hdr:
            failures.append("stub kept for a supported API")
        if "acblasAsumEx_64" not in hdr:
            failures.append("stub dropped for an unsupported API")
        hdr_off = G.generate_stubs_header(stub_probe=False)
        if "acblasSetKernelStream" not in hdr_off or "PPU_SDK_VERSION" not in hdr_off:
            failures.append("probe off does not restore gated stubs")
    return failures


def test_compat_version_mismatch_fallback(tmp):
    failures = []
    sdk = _make_sdk(tmp)
    wrap = _make_wrapper(tmp, sdk)
    from sailify.sailify_python import write_compat_wrapper_header
    with _Env(PATH=wrap + ":/usr/bin:/bin", PPU_PATH=sdk):
        compat = os.path.join(tmp, "compat_match")
        write_compat_wrapper_header(compat, ppu_sdk_version=(2, 2, 0))
        hdr = open(os.path.join(compat, "ppu_unsupported_stubs.h")).read()
        if "acblasSetKernelStream" in hdr:
            failures.append("matching version still emits a supported API's stub")
        compat = os.path.join(tmp, "compat_mismatch")
        write_compat_wrapper_header(compat, ppu_sdk_version=(2, 3))
        hdr = open(os.path.join(compat, "ppu_unsupported_stubs.h")).read()
        if "acblasSetKernelStream" not in hdr:
            failures.append("mismatched version drops a stub instead of gating it")
        if "#if PPU_SDK_VERSION" not in hdr:
            failures.append("mismatched version keeps probe mode without gates")
    return failures


def main(argv=None):
    tmp = tempfile.mkdtemp(prefix="sailify_probe_")
    npass = 0
    nfail = 0
    try:
        for name, fn in [
            ("sdk_dir_via_wrapper_and_env", test_sdk_dir_via_wrapper_and_env),
            ("sdk_dir_via_symlinked_hgcc", test_sdk_dir_via_symlinked_hgcc),
            ("sdk_dir_rejects_non_sdk_dirs", test_sdk_dir_rejects_non_sdk_dirs),
            ("sdk_dir_rejects_version_mismatch", test_sdk_dir_rejects_version_mismatch),
            ("resolve_host_per_candidate", test_resolve_host_per_candidate),
            ("symbols_strip_version_suffix", test_symbols_strip_version_suffix),
            ("probe_supported_apis", test_probe_supported_apis),
            ("compat_version_mismatch_fallback", test_compat_version_mismatch_fallback),
        ]:
            try:
                failures = fn(tmp)
            except Exception as exc:
                failures = ["exception: %r" % (exc,)]
            if failures:
                nfail += 1
                for f in failures:
                    print("FAIL probe " + name + ": " + f)
            else:
                npass += 1
                print("PASS probe " + name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _report(npass, nfail)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

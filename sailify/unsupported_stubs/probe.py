"""Probe the PPU SDK's shared libraries for exported API symbols.

Symbols come from the defined dynamic symbol table of each host library
(``nm -D --defined-only``), plus a demangling pass for C++ names.  The SDK
is located with the same priority as the version detection (hgcc on PATH,
else ``$PPU_SDK``/``$PPU_HOME``, else ``$PPU_PATH``), with the compiler
symlink resolved so both look at the same — and only one — installation.
"""

import os
import re
import logging
import subprocess

from ..ppu_sdk_version import _find_hgcc

log = logging.getLogger(__name__)

__all__ = ["probe_supported_apis", "find_sdk_lib_dir"]

# Extra env vars that may point at the SDK root (checked after hgcc).
_LIB_DIR_ENV_VARS = ("PPU_PATH", "PPU_SDK", "PPU_HOME")

# stub JSON "library" key -> candidate .so basenames under <sdk>/lib.
# Libraries without a host .so (cufftw, cufile) are probe-unavailable and
# keep their stubs.
_LIBRARY_HOSTS = {
    "cublas": ["libacblas.so"],
    "cublasLt": ["libacblasLt.so"],
    "cudnn": ["libacdnn.so"],
    "cufft": ["libacfft.so"],
    "cufftw": [],
    "curand": ["libacrand.so"],
    "cusolver": ["libacsolver.so"],
    "cusparse": ["libacsparse.so"],
    "nvrtc": ["libhgrtc.so"],
    "nvidia-ml": ["libhgml.so"],
    "cupti": ["libhgpti.so"],
    "cuda": [
        "libhggcrt1.so",
        "libhggc.so",
        "libhggc_wrapper.so",
        "libaccpu.so",
    ],
    "cudart": [
        "libhggcrt1.so",
        "libhggc.so",
        "libhggc_wrapper.so",
        "libaccpu.so",
    ],
    "cufile": [],
}

# Identifier that directly precedes a '(' in a demangled symbol, e.g.
# "hgptiFoo::bar(int)" -> "bar" and "hgptiFoo(int)" -> "hgptiFoo".
_DEMANGLED_FUNC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def find_sdk_lib_dir():
    """The single SDK lib directory to probe, or None.

    Prefers the SDK of the selected compiler — realpath, so a symlinked hgcc
    (e.g. ``/usr/local/bin/hgcc -> /opt/sdk/bin/hgcc``) resolves to the real
    installation.  Falls back to the environment variables only when the
    compiler is not found or lives outside an SDK (e.g. a wrapper script).
    Probing never mixes installations: a host library missing in this
    directory keeps the legacy version-gated stub.
    """
    hgcc = _find_hgcc()
    if hgcc:
        cand = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(hgcc)), "..", "lib")
        )
        if os.path.isdir(cand):
            return cand
    for var in _LIB_DIR_ENV_VARS:
        home = os.environ.get(var)
        if home:
            cand = os.path.join(home, "lib")
            if os.path.isdir(cand):
                return cand
    return None


def _nm_symbols(path, demangle):
    """Full symbol strings (``addr type name...`` minus addr and type).

    Demangled names contain spaces ("foo(unsigned char)"), so the name part
    is everything from the third field on.  Returns None on any nm failure.
    """
    cmd = ["nm", "-D", "--defined-only"]
    if demangle:
        cmd.append("-C")
    cmd.append(path)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Failed to run nm on %s: %s", path, exc)
        return None
    if proc.returncode != 0:
        log.warning(
            "nm exited %d on %s: %s", proc.returncode, path,
            (proc.stderr or "")[:200],
        )
        return None
    return [" ".join(line.split()[2:]) for line in proc.stdout.splitlines()
            if len(line.split()) >= 3]


def _symbols_of_so(path):
    """Symbol names exported by *path*; None when nm cannot read it.

    Raw symbol names are used as-is (minus any '@version' suffix); from
    demangled entries, identifiers directly preceding a '(' are collected.
    """
    raw = _nm_symbols(path, demangle=False)
    demangled = _nm_symbols(path, demangle=True)
    if raw is None or demangled is None:
        return None
    names = set()
    for sym in raw:
        names.add(sym.split("@")[0])
    for sym in demangled:
        for m in _DEMANGLED_FUNC_RE.finditer(sym):
            names.add(m.group(1))
    return names


def _resolve_host(lib_dir, so_names):
    if lib_dir and os.path.isfile(os.path.join(lib_dir, so_names[0])):
        return [os.path.join(lib_dir, n) for n in so_names
                if os.path.isfile(os.path.join(lib_dir, n))]
    return []


class ProbeResult:
    """Outcome of one SDK probe.

    ``supported`` holds names exported by a probed host library, ``probed``
    the library keys whose host libraries were fully scanned, and ``unknown``
    the library keys that could not be probed — their stubs keep the legacy
    version gates.
    """

    def __init__(self):
        self.supported = set()
        self.probed = set()
        self.unknown = []
        self.scanned = []


def probe_supported_apis(apis):
    """Probe which of *apis* (dicts from unsupported_apis.json) are supported.

    A library counts as probed only when all of its host libraries were read
    successfully; otherwise it lands in ``unknown`` and the caller is
    expected to keep its version-gated legacy stubs.
    """
    result = ProbeResult()
    lib_dir = find_sdk_lib_dir()
    if lib_dir is None:
        log.warning(
            "stub probe: no PPU SDK lib directory found (set $PPU_PATH/"
            "$PPU_SDK/$PPU_HOME or put hgcc on PATH); keeping the legacy "
            "stub set",
        )
        return result

    by_lib = {}
    for api in apis:
        by_lib.setdefault(api.get("library", ""), []).append(api["name"])

    host_cache = {}
    symbols_cache = {}
    for lib, names in sorted(by_lib.items()):
        so_names = _LIBRARY_HOSTS.get(lib)
        if so_names is None:
            log.warning("stub probe: unknown library %r, keeping its stubs", lib)
            result.unknown.append(lib)
            continue
        if not so_names:
            result.unknown.append(lib)
            continue
        if tuple(so_names) not in host_cache:
            host_cache[tuple(so_names)] = _resolve_host(lib_dir, so_names)
        hosts = host_cache[tuple(so_names)]
        if not hosts:
            result.unknown.append(lib)
            continue
        symbols = set()
        failed = False
        for host in hosts:
            if host not in symbols_cache:
                result.scanned.append(host)
                symbols_cache[host] = _symbols_of_so(host)
            syms = symbols_cache[host]
            if syms is None:
                failed = True
            else:
                symbols |= syms
        if failed:
            log.warning(
                "stub probe: could not read symbols for library %r; keeping "
                "its stubs with their version gates", lib,
            )
            result.unknown.append(lib)
            continue
        result.probed.add(lib)
        result.supported.update(n for n in names if n in symbols)

    return result

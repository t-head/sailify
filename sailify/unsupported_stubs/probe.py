"""Probe the PPU SDK's shared libraries for exported API symbols.

Symbols come from ``nm -D --defined-only`` on each host library.  The SDK
is located like the version detection: hgcc on PATH (symlink resolved),
else ``$PPU_PATH``/``$PPU_SDK``/``$PPU_HOME`` — only one installation.
"""

import os
import logging
import subprocess

from ..ppu_sdk_version import _HGCC_VERSION_RE, _find_hgcc

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

_ALL_HOST_SOS = {so for sos in _LIBRARY_HOSTS.values() for so in sos}


def _is_sdk_lib_dir(path):
    """True when *path* contains at least one known PPU host library.

    Existence alone is not enough: with hgcc a wrapper script in e.g.
    ``/usr/local/bin``, ``../lib`` would resolve to the unrelated
    ``/usr/local/lib`` and silently hijack the probe.
    """
    try:
        return any(n in _ALL_HOST_SOS for n in os.listdir(path))
    except OSError:
        return False


def _hgcc_version(hgcc_path):
    """Normalized ``(major, minor, patch)`` of ``Release version X.Y[.Z]``, or None."""
    try:
        proc = subprocess.run(
            [hgcc_path, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _HGCC_VERSION_RE.search(proc.stdout or "")
    if not m:
        return None
    major, minor, patch = m.groups()
    return (int(major), int(minor), int(patch) if patch else 0)


def find_sdk_lib_dir():
    """The single SDK lib directory to probe, or None.

    Prefers the selected compiler's SDK (realpath, so a symlinked hgcc
    resolves to the real installation) and falls back to the environment
    variables when the compiler is missing or lives outside an SDK.  A
    candidate must contain a PPU host library; probing never mixes
    installations.
    """
    hgcc = _find_hgcc()
    if hgcc:
        cand = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(hgcc)), "..", "lib")
        )
        if _is_sdk_lib_dir(cand):
            return cand
    for var in _LIB_DIR_ENV_VARS:
        home = os.environ.get(var)
        if not home:
            continue
        cand = os.path.join(home, "lib")
        if not _is_sdk_lib_dir(cand):
            continue
        # hgcc on PATH (e.g. a wrapper) may belong to a different SDK than
        # the environment points at; the probe must describe the SDK whose
        # version is baked into the header, so require a version match.
        # An unreadable version on either side cannot prove the match.
        if hgcc:
            selected = _hgcc_version(os.path.join(cand, "..", "bin", "hgcc"))
            current = _hgcc_version(hgcc)
            if selected is None or current is None or selected != current:
                log.warning(
                    "stub probe: %s points at a different SDK than %s; keeping "
                    "the legacy stub set", var, hgcc,
                )
                return None
        return cand
    return None


def _nm_symbols(path):
    """Full symbol strings (``addr type name...`` minus addr and type), or None on nm failure."""
    cmd = ["nm", "-D", "--defined-only", path]
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
    """Symbol names exported by *path* (minus '@version' suffixes), or None on nm failure.

    Only raw names are matched: every unsupported_apis.json entry is a C
    API, and demangled C++ entries (e.g. ``internal::foo``) would be
    mistaken for the like-named global C API.
    """
    raw = _nm_symbols(path)
    if raw is None:
        return None
    return {sym.split("@")[0] for sym in raw}


def _resolve_host(lib_dir, so_names):
    """All candidates existing in *lib_dir* — each checked independently."""
    if not lib_dir:
        return []
    return [os.path.join(lib_dir, n) for n in so_names
            if os.path.isfile(os.path.join(lib_dir, n))]


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

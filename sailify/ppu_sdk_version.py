"""PPU SDK version detection for version-conditional fixups.

The version is resolved once per conversion and baked into
``compatible_wrapper.h`` as ``PPU_SDK_VERSION``
(``major * 10000 + minor * 100 + patch``; 0 = unknown keeps the
legacy behavior).
"""

import os
import re
import shutil
import logging
import subprocess

__all__ = [
    "DEFAULT_RUNTIME_API_VERSION",
    "detect_ppu_sdk_version",
    "detect_runtime_api_version",
    "parse_version_string",
    "format_version_string",
    "version_to_int",
]

log = logging.getLogger(__name__)

_SDK_ENV_VARS = ("PPU_SDK", "PPU_HOME")
_VERSION_TXT_ENV_VARS = ("PPU_PATH",) + _SDK_ENV_VARS

_HGCC_VERSION_RE = re.compile(r"Release version (\d+)\.(\d+)(?:\.(\d+))?")
_RUNTIME_API_RE = re.compile(r"Runtime API version\s+v\s*(\d+)", re.IGNORECASE)
_HGGCRT_VERSION_RE = re.compile(r"hggcrt_version\s*[:=]\s*v?(\d+)", re.IGNORECASE)

DEFAULT_RUNTIME_API_VERSION = 3

_hgcc_version_cache = None
_hgcc_version_queried = False


def parse_version_string(version_str):
    """Parse an explicit version like "2.2", "2v2", "2_2" or "2.1.1"
    into a (major, minor, patch) tuple (patch defaults to 0)."""
    m = re.match(
        r"^\s*(\d+)(?:\s*[v._]\s*(\d+))?(?:\s*[v._]\s*(\d+))?\s*$",
        str(version_str),
    )
    if not m:
        raise ValueError(
            f"Invalid PPU SDK version '{version_str}': expected e.g. '2.1', "
            "'2v2' or '2.1.1'."
        )
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def _as_3_tuple(version):
    padded = tuple(version) + (0, 0, 0)
    return padded[0], padded[1], padded[2]


def version_to_int(version):
    """Encode a (major, minor, patch) tuple as the PPU_SDK_VERSION macro value."""
    if not version:
        return 0
    major, minor, patch = _as_3_tuple(version)
    return major * 10000 + minor * 100 + patch


def format_version_string(version):
    """Render a version tuple as "X.Y" (patch 0) or "X.Y.Z"; None -> "unknown"."""
    if not version:
        return "unknown"
    major, minor, patch = _as_3_tuple(version)
    return f"{major}.{minor}" if patch == 0 else f"{major}.{minor}.{patch}"


def _find_hgcc():
    """Locate the hgcc compiler for the build."""
    hgcc = shutil.which("hgcc")
    if hgcc:
        return hgcc
    for var in _SDK_ENV_VARS:
        home = os.environ.get(var)
        if home:
            candidate = os.path.join(home, "bin", "hgcc")
            if os.path.isfile(candidate):
                return candidate
    return None


def _query_hgcc_version():
    global _hgcc_version_cache, _hgcc_version_queried
    if _hgcc_version_queried:
        return _hgcc_version_cache
    _hgcc_version_queried = True
    hgcc = _find_hgcc()
    if not hgcc:
        return None
    try:
        proc = subprocess.run(
            [hgcc, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Failed to run %s --version: %s", hgcc, exc)
        _hgcc_version_cache = None
        return None
    _hgcc_version_cache = (proc.stdout or "", proc.stderr or "")
    return _hgcc_version_cache


def detect_ppu_sdk_version():
    """Detect the PPU SDK version from ``hgcc --version``;
    (major, minor, patch) or None."""
    hgcc = _find_hgcc()
    if not hgcc:
        log.warning(
            "Could not find the hgcc compiler (put it in PATH or set $%s/"
            "$PPU_HOME, or pass --ppu-sdk-version); fixups keep the legacy "
            "behavior.",
            _SDK_ENV_VARS[0],
        )
        return None
    output = _query_hgcc_version()
    if output is None:
        return None
    stdout, stderr = output
    m = _HGCC_VERSION_RE.search(stdout)
    if not m:
        log.warning(
            "Could not parse 'Release version X.Y[.Z]' from %s --version output: %r",
            hgcc, (stdout or stderr)[:200],
        )
        return None
    version = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
    log.info("Detected PPU SDK version %s via %s --version",
             format_version_string(version), hgcc)
    return version


def _runtime_api_version_from_version_txt():
    for var in _VERSION_TXT_ENV_VARS:
        home = os.environ.get(var)
        if not home:
            continue
        path = os.path.join(home, "VERSION.txt")
        try:
            with open(path, "r", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        m = _HGGCRT_VERSION_RE.search(content)
        if m:
            version = int(m.group(1))
            log.info("Detected hggcrt Runtime API version v%d via %s",
                     version, path)
            return version
    return None


def detect_runtime_api_version():
    output = _query_hgcc_version()
    if output is not None:
        stdout, _stderr = output
        m = _RUNTIME_API_RE.search(stdout)
        if m:
            version = int(m.group(1))
            log.info("Detected hggcrt Runtime API version v%d via hgcc --version",
                     version)
            return version
    version = _runtime_api_version_from_version_txt()
    if version is not None:
        return version
    log.info("hggcrt Runtime API version not advertised by hgcc or VERSION.txt; "
             "assuming v%d", DEFAULT_RUNTIME_API_VERSION)
    return DEFAULT_RUNTIME_API_VERSION

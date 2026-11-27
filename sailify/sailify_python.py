"""
Core CUDA → PPU conversion engine.

Implements:
  - File discovery (extension-based filtering)
  - Trie-based regex substitution for CUDA identifiers
  - Header include-path rewriting
  - Custom and extra mapping support
  - Compatibility header generation

sailify only converts CUDA SDK identifiers (APIs, types, macros, header paths).
It does NOT rename files or directories, and does NOT rewrite CMake or Python
build scripts.
"""

import os
import re
import json
import shutil
import logging
import fnmatch
from bisect import bisect_right
from enum import Enum
from typing import Iterable, Iterator, Optional, Dict, List

from .cuda_to_ppu_mappings import (
    CUDA_TO_PPU_MAPPINGS,
    PPU_SDK_VERSION_MAPPINGS,
)
from .unsupported_stubs import write_stubs_header
from .ppu_sdk_version import (
    detect_ppu_sdk_version,
    detect_runtime_api_version,
    format_version_string,
    version_to_int,
)

__all__ = [
    "CurrentState",
    "SailifyResult",
    "Trie",
    "COMPAT_WRAPPER_INCLUDE_LINE",
    "compat_wrapper_include_line",
    "update_custom_mappings",
    "load_extra_mappings",
    "matched_files_iter",
    "get_ppu_file_path",
    "preprocessor",
    "sailify",
    "sailify_extra_files",
    "sailify_extra_files_recursive",
    "write_compat_wrapper_header",
]

log = logging.getLogger(__name__)

# ── Default version configuration ─────────────────────────────────────
# Default versions for the COMPATIBLE_XXX macros in compatible_wrapper.h.
# These can be overridden via CLI (--cuda-version, etc.) or --config-json.
DEFAULT_VERSION_CONFIG_V3 = {
    "cuda": "13.0.0",
    "cublas": "13.0.0",
    "cufft": "12.0.0",
    "curand": "10.4.0",
    "cusparse": "12.0.0",
    "cusolver": "12.0.3",
    "cudnn": "8.9.5",
    "nccl": "2.27.3",
    "video": "13.0.19",
    "cupti": 130000,
    "npp": "13.0.50",
}

DEFAULT_VERSION_CONFIG_V2 = {
    "cuda": "12.9.0",
    "cublas": "12.9.0",
    "cufft": "11.4.0",
    "curand": "10.3.10",
    "cusparse": "12.5.9",
    "cusolver": "11.7.4",
    "cudnn": "8.9.5",
    "nccl": "2.27.3",
    "video": "13.0.19",
    "cupti": 120900,
    "npp": "12.4.27",
}


def _default_version_config(runtime_api_version):
    if runtime_api_version == 2:
        return DEFAULT_VERSION_CONFIG_V2
    return DEFAULT_VERSION_CONFIG_V3


def _parse_version_string(version_str, encoding="cuda"):
    """Parse a version string like '13.0.0' into the encoded integer value."""
    if encoding == "raw":
        try:
            return int(version_str)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid version value '{version_str}' for encoding 'raw': "
                f"expected an integer. {e}"
            )
    parts = str(version_str).split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid version string '{version_str}': version components must be "
            f"integers separated by dots (e.g. '13.0.0'). {e}"
        )
    if encoding == "cuda":
        return major * 1000 + minor * 10
    elif encoding == "cublas":
        return major * 10000 + minor * 100 + patch
    elif encoding in ("cufft", "curand", "cusparse", "cusolver", "cudnn"):
        return major * 1000 + minor * 100 + patch
    elif encoding == "nccl":
        return major * 10000 + minor * 100 + patch
    elif encoding == "video":
        return major * 1000 + minor * 10 + patch
    else:
        raise ValueError(f"Unknown version encoding: {encoding}")


def _generate_compat_wrapper_header(version_config=None, ppu_sdk_version=None,
                                    runtime_api_version=None):
    """Generate the content of compatible_wrapper.h with the given version config.

    *ppu_sdk_version* is a (major, minor[, patch]) tuple of the PPU SDK
    (detected by sailify or set via --ppu-sdk-version).  None means unknown
    and yields PPU_SDK_VERSION 0, i.e. the legacy fixup behavior.
    """
    cfg = dict(_default_version_config(runtime_api_version))
    if version_config:
        cfg.update(version_config)

    cuda_ver = _parse_version_string(cfg["cuda"], "cuda")
    cudnn_ver = _parse_version_string(cfg["cudnn"], "cudnn")
    nccl_ver = _parse_version_string(cfg["nccl"], "nccl")
    video_ver = _parse_version_string(cfg["video"], "video")
    cupti_ver = int(cfg["cupti"]) if not isinstance(cfg["cupti"], int) else cfg["cupti"]

    cublas_ver = _parse_version_string(cfg["cublas"], "cublas")
    cufft_ver = _parse_version_string(cfg["cufft"], "cufft")
    curand_ver = _parse_version_string(cfg["curand"], "curand")
    cusparse_ver = _parse_version_string(cfg["cusparse"], "cusparse")
    cusolver_ver = _parse_version_string(cfg["cusolver"], "cusolver")

    # Parse major/minor/patch components for COMPATIBLE_*_MAJOR etc.
    def _ver_parts(s):
        p = str(s).split(".")
        return (int(p[0]) if len(p) > 0 else 0,
                int(p[1]) if len(p) > 1 else 0,
                int(p[2]) if len(p) > 2 else 0)
    cuda_major, cuda_minor, cuda_patch = _ver_parts(cfg["cuda"])
    cudnn_major, cudnn_minor, cudnn_patch = _ver_parts(cfg["cudnn"])
    cublas_major, cublas_minor, cublas_patch = _ver_parts(cfg["cublas"])
    cufft_major, cufft_minor, cufft_patch = _ver_parts(cfg["cufft"])
    curand_major, curand_minor, curand_patch = _ver_parts(cfg["curand"])
    cusparse_major, cusparse_minor, cusparse_patch = _ver_parts(cfg["cusparse"])
    cusolver_major, cusolver_minor, cusolver_patch = _ver_parts(cfg["cusolver"])
    nccl_major, nccl_minor, nccl_patch = _ver_parts(cfg["nccl"])
    npp_major, npp_minor, npp_build = _ver_parts(cfg["npp"])

    if ppu_sdk_version:
        ppu_sdk_major, ppu_sdk_minor, ppu_sdk_patch = \
            (tuple(ppu_sdk_version) + (0, 0, 0))[:3]
        ppu_sdk_str = format_version_string(ppu_sdk_version)
    else:
        ppu_sdk_major, ppu_sdk_minor, ppu_sdk_patch = 0, 0, 0
        ppu_sdk_str = "unknown"

    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compatible_wrapper.h.tpl")
    with open(tpl_path, "r") as f:
        template = f.read()

    return template.format(
        cuda_str=cfg["cuda"],
        cuda_ver=cuda_ver,
        cuda_major=cuda_major, cuda_minor=cuda_minor, cuda_patch=cuda_patch,
        cudnn_str=cfg["cudnn"],
        cudnn_ver=cudnn_ver,
        cudnn_major=cudnn_major, cudnn_minor=cudnn_minor, cudnn_patch=cudnn_patch,
        cublas_str=cfg["cublas"],
        cublas_ver=cublas_ver,
        cublas_major=cublas_major, cublas_minor=cublas_minor, cublas_patch=cublas_patch,
        cufft_str=cfg["cufft"],
        cufft_ver=cufft_ver,
        cufft_major=cufft_major, cufft_minor=cufft_minor, cufft_patch=cufft_patch,
        curand_str=cfg["curand"],
        curand_ver=curand_ver,
        curand_major=curand_major, curand_minor=curand_minor, curand_patch=curand_patch,
        cusparse_str=cfg["cusparse"],
        cusparse_ver=cusparse_ver,
        cusparse_major=cusparse_major, cusparse_minor=cusparse_minor, cusparse_patch=cusparse_patch,
        cusolver_str=cfg["cusolver"],
        cusolver_ver=cusolver_ver,
        cusolver_major=cusolver_major, cusolver_minor=cusolver_minor, cusolver_patch=cusolver_patch,
        nccl_str=cfg["nccl"],
        nccl_ver=nccl_ver,
        nccl_major=nccl_major, nccl_minor=nccl_minor, nccl_patch=nccl_patch,
        video_str=cfg["video"],
        video_ver=video_ver,
        cupti_ver=cupti_ver,
        npp_major=npp_major, npp_minor=npp_minor, npp_build=npp_build,
        ppu_sdk_str=ppu_sdk_str,
        ppu_sdk_major=ppu_sdk_major,
        ppu_sdk_minor=ppu_sdk_minor,
        ppu_sdk_patch=ppu_sdk_patch,
    )


# Static compat headers that ship alongside compatible_wrapper.h
_COMPAT_STATIC_HEADERS = ["hgperf_common.h", "hgperf_host.h", "ppu_sdk_fixups.h", "mma.h", "hggc_fp4.h", "hggc_device_runtime_api.h"]

_SDK_2_3_VERSION_INT = 20300

_SDK23_SHIM_HEADERS = [
    ("hggc_channel_descriptor.h", "channel_descriptor.h"),
    ("hggc_driver_functions.h", "driver_functions.h"),
    ("hggc_driver_types.h", "driver_types.h"),
    ("hggc_host_defines.h", "host_defines.h"),
    ("hggc_vector_functions.h", "vector_functions.h"),
    ("hggc_vector_functions.hpp", "vector_functions.hpp"),
    ("hggc_vector_types.h", "vector_types.h"),
    ("hggc_texture_types.h", "texture_types.h"),
    ("hggc_sanitizer.h", "sanitizer.h"),
    ("hggc_sanitizer_barrier.h", "sanitizer_barrier.h"),
    ("hggc_sanitizer_callbacks.h", "sanitizer_callbacks.h"),
    ("hggc_sanitizer_driver_cbid.h", "sanitizer_driver_cbid.h"),
    ("hggc_sanitizer_memory.h", "sanitizer_memory.h"),
    ("hggc_sanitizer_patching.h", "sanitizer_patching.h"),
    ("hggc_sanitizer_result.h", "sanitizer_result.h"),
    ("hggc_sanitizer_runtime_cbid.h", "sanitizer_runtime_cbid.h"),
    ("hggc_sanitizer_stream.h", "sanitizer_stream.h"),
    ("hggc_surface_types.h", "surface_types.h"),
]


def _generate_sdk_2_3_shim(name, fallback):
    """Shim header for the PPU_SDK 2v3 renames: pass through to the SDK
    header with the same name when it exists, otherwise forward to
    *fallback* (the pre-rename spelling)."""
    return (
        f"#if __has_include_next(<{name}>)\n"
        f"#include_next <{name}>\n"
        "#else\n"
        f"#include <{fallback}>\n"
        "#endif\n"
    )


def write_compat_wrapper_header(compat_dir, version_config=None,
                                ppu_sdk_version=None, runtime_api_version=None,
                                stub_probe=True):
    """Write compatible_wrapper.h and static compat headers to compat_dir/."""
    os.makedirs(compat_dir, exist_ok=True)
    header_path = os.path.join(compat_dir, "compatible_wrapper.h")

    content = _generate_compat_wrapper_header(
        version_config, ppu_sdk_version, runtime_api_version
    )
    with open(header_path, "w") as f:
        f.write(content)
    if version_config:
        log.info("Generated custom compatible_wrapper.h with version config: %s", version_config)
    else:
        log.info("Generated default compatible_wrapper.h in %s", compat_dir)

    if stub_probe and ppu_sdk_version is not None:
        detected = detect_ppu_sdk_version()
        if detected is None or version_to_int(detected) != version_to_int(ppu_sdk_version):
            log.warning(
                "stub probe: requested PPU SDK %s differs from the installed %s; "
                "falling back to version-gated legacy stubs",
                format_version_string(ppu_sdk_version),
                format_version_string(detected) if detected else "unknown",
            )
            stub_probe = False

    write_stubs_header(compat_dir, stub_probe=stub_probe)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for hdr in _COMPAT_STATIC_HEADERS:
        src = os.path.join(pkg_dir, hdr)
        if os.path.isfile(src):
            dst = os.path.join(compat_dir, hdr)
            shutil.copy2(src, dst)
            log.info("Copied %s to %s", hdr, compat_dir)

    if ppu_sdk_version is not None and version_to_int(ppu_sdk_version) >= _SDK_2_3_VERSION_INT:
        for name, fallback in _SDK23_SHIM_HEADERS:
            with open(os.path.join(compat_dir, name), "w") as f:
                f.write(_generate_sdk_2_3_shim(name, fallback))


# ── File extensions ────────────────────────────────────────────────────
DEFAULT_EXTENSIONS = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".inl")
DEFAULT_HEADER_EXTENSIONS = (".cuh", ".h", ".hpp")

SKIP_DIRS = {".git", "build", "__pycache__", ".tox", "node_modules", ".ppu_compat"}

# Prepended to converted headers when inject_compat_include is enabled; resolves
# via the include search path pointing at the PARENT of the compat dir.
COMPAT_WRAPPER_INCLUDE_LINE = "#include <.ppu_compat/compatible_wrapper.h>"


def compat_wrapper_include_line(compat_dir: str) -> str:
    """Wrapper include line for *compat_dir* (its basename is the path segment)."""
    name = os.path.basename(os.path.normpath(compat_dir))
    return f"#include <{name}/compatible_wrapper.h>"


_COMPAT_INCLUDE_LINE_RE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*<[^\r\n>]*compatible_wrapper\.h>[ \t]*\r?\n?",
    re.MULTILINE,
)


def _inject_compat_include(source: str, include_line: str = COMPAT_WRAPPER_INCLUDE_LINE) -> str:
    """Prepend *include_line*, replacing any wrapper include left by an earlier run."""
    has_bom = source.startswith("\ufeff")
    body = source[1:] if has_bom else source
    body = _COMPAT_INCLUDE_LINE_RE.sub("", body)
    return ("\ufeff" if has_bom else "") + include_line + "\n" + body

class CurrentState(Enum):
    INITIALIZED = 1
    DONE = 2


class SailifyResult:
    """Result of processing a single file."""

    def __init__(self, current_state: CurrentState, ppuified_path: str):
        self.current_state = current_state
        self.ppuified_path = ppuified_path
        self.status = ""

    def asdict(self):
        return {"ppuified_path": self.ppuified_path, "status": self.status}


# ── Trie ──────────────────────────────────────────────────────────────

class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = _TrieNode()

    def add(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def _pattern(self, node: _TrieNode) -> str:
        """Recursively build the regex sub-pattern for *node*."""
        if not node.children:
            return ""

        leaves = []
        branches = []
        for ch in sorted(node.children):
            child = node.children[ch]
            sub = self._pattern(child)
            if sub:
                branches.append(re.escape(ch) + sub)
            else:
                leaves.append(re.escape(ch))

        parts = []
        if leaves:
            if len(leaves) == 1:
                parts.append(leaves[0])
            else:
                parts.append("[" + "".join(leaves) + "]")
        parts.extend(branches)

        if node.is_end:
            if len(parts) == 1:
                p = parts[0]
                if len(p) == 1 or (p.startswith("[") and p.endswith("]")):
                    return p + "?"
                return "(?:" + p + ")?"
            return "(?:" + "|".join(parts) + ")?"
        if len(parts) == 1:
            return parts[0]
        return "(?:" + "|".join(parts) + ")"

    def export_to_regex(self) -> str:
        """Return a regex string that matches any word in the trie."""
        return self._pattern(self.root)


# ── Global trie / map / compiled regex ────────────────────────────────

_PPU_TRIE = Trie()
_PPU_MAP: Dict[str, str] = {}

for _mapping in CUDA_TO_PPU_MAPPINGS:
    for _src, _dst in _mapping.items():
        _PPU_TRIE.add(_src)
        _PPU_MAP[_src] = _dst

for _kind, _min_version, _mapping in PPU_SDK_VERSION_MAPPINGS:
    if _kind == "identifier":
        for _src in _mapping:
            _PPU_TRIE.add(_src)
            # Base-map consumers have no SDK selection.
            _PPU_MAP.setdefault(_src, _src)

_RE_PPU_PREPROCESSOR = re.compile(
    r"(?<!\w)(" + _PPU_TRIE.export_to_regex() + r")(?!\w)"
)


def _get_versioned_mappings(kind, ppu_sdk_version):
    version = version_to_int(ppu_sdk_version)
    mappings = {}
    for mapping_kind, min_version, mapping in PPU_SDK_VERSION_MAPPINGS:
        if mapping_kind == kind and version >= version_to_int(min_version):
            mappings.update(mapping)
    return mappings


# ── Custom mapping support ────────────────────────────────────────────

_CUSTOM_TRIE = Trie()
_CUSTOM_SPECIAL_MAP: Dict[str, str] = {}


def update_custom_mappings(custom_map_list: str) -> None:
    """Load additional CUDA→PPU mappings from a JSON file."""
    global _CUSTOM_TRIE, _CUSTOM_SPECIAL_MAP
    if not custom_map_list or not os.path.isfile(custom_map_list):
        return
    try:
        with open(custom_map_list) as fh:
            custom_map = json.load(fh)
        if not isinstance(custom_map, dict):
            log.warning("Custom map file must contain a JSON object, ignoring")
            return
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load custom mappings: {e}")
        return
    _CUSTOM_TRIE = Trie()
    _CUSTOM_SPECIAL_MAP = {}
    for src, dst in custom_map.items():
        _CUSTOM_TRIE.add(src)
        _CUSTOM_SPECIAL_MAP[src] = dst


# ── Extra mapping lists (from --extra-mapping config) ────────────────
# A list of (pattern, replacement) string pairs applied AFTER the main
# trie replacement step.  Loaded from a JSON file via --extra-mapping.
_EXTRA_MAPPING_LISTS = []


def load_extra_mappings(extra_mapping_path: str) -> None:
    """Load extra string-to-string mappings from a JSON file.
    Each pair is applied via ``str.replace`` on the post-trie source text.
    """
    global _EXTRA_MAPPING_LISTS
    if not extra_mapping_path or not os.path.isfile(extra_mapping_path):
        return
    try:
        with open(extra_mapping_path) as fh:
            mappings = json.load(fh)
        if not isinstance(mappings, list):
            log.warning("Extra mapping file must contain a JSON array, ignoring")
            return
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load extra mappings: {e}")
        return
    _EXTRA_MAPPING_LISTS = [(str(src), str(dst)) for src, dst in mappings]
    log.info(f"Loaded {len(_EXTRA_MAPPING_LISTS)} extra mappings from {extra_mapping_path}")


# ── File discovery ────────────────────────────────────────────────────

def _matches_pattern(relpath: str, pattern: str) -> bool:
    """Check if a relative path matches a glob pattern."""
    rel_normalized = relpath.replace(os.sep, "/")
    pat_normalized = pattern.replace(os.sep, "/")

    if "**" in pat_normalized:
        regex_parts = []
        parts = pat_normalized.split("**")
        for i, part in enumerate(parts):
            if part:
                escaped = re.escape(part)
                escaped = escaped.replace(r"\*", "[^/]*")
                escaped = escaped.replace(r"\?", "[^/]")
                regex_parts.append(escaped)
            else:
                regex_parts.append("")

        regex = ".*".join(regex_parts)

        if not pat_normalized.startswith("**"):
            regex = "^" + regex
        if not pat_normalized.endswith("**"):
            regex = regex + "$"
        else:
            regex = regex.rstrip("$") + ".*$"

        return bool(re.match(regex, rel_normalized))
    else:
        return fnmatch.fnmatch(rel_normalized, pat_normalized)


def _should_include(relpath: str, includes: Iterable[str], ignores: Iterable[str]) -> bool:
    """Return True if *relpath* matches *includes* and does not match *ignores*."""
    for pat in ignores:
        if _matches_pattern(relpath, pat):
            return False
    if not includes:
        return True
    for pat in includes:
        if _matches_pattern(relpath, pat):
            return True
    return False


def matched_files_iter(
    root_path: str,
    includes: Iterable[str] = ("*",),
    ignores: Iterable[str] = (),
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    excludes: Iterable[str] = (),
) -> Iterator[str]:
    """Yield absolute paths of files under *root_path* that should be processed."""
    ext_set = set(extensions)
    # Normalize excludes: separate basename-only from relative path excludes
    _rel_excludes = []
    for exc in excludes:
        normalized = exc.replace("\\", "/").strip("/")
        if "/" in normalized:
            _rel_excludes.append(normalized)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # Prune relative path excludes
        if _rel_excludes:
            try:
                rel_root = os.path.relpath(dirpath, root_path).replace(os.sep, "/")
            except ValueError:
                rel_root = ""
            filtered_dirs = []
            for d in dirnames:
                child_rel = (rel_root + "/" + d) if rel_root and rel_root != "." else d
                skip = False
                for exc in _rel_excludes:
                    if child_rel == exc or child_rel.startswith(exc + "/"):
                        skip = True
                        break
                if not skip:
                    filtered_dirs.append(d)
            dirnames[:] = filtered_dirs
        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext not in ext_set:
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_path)
            if not _should_include(rel, includes, ignores):
                continue
            yield full


# ── Path rewriting ────────────────────────────────────────────────────

def get_ppu_file_path(rel_filepath: str) -> str:
    """Compute the PPU-equivalent output path for a CUDA source file.

    sailify does NOT rename files or directories — the output path is the
    same as the input path. This function is kept for API compatibility.
    """
    return rel_filepath


def _mask_comments_and_strings(source: str, mask_includes: bool = False,
                               mask_string_contents: bool = False) -> str:
    """Mask comments and char literals with 'x' to gate identifier replacement."""
    include_ends = {
        start: end for start, end, _path in _iter_header_names(source)
    } if mask_includes else {}
    n = len(source)
    in_include = [False] * n
    line_start = 0
    while line_start < n:
        line_end = source.find('\n', line_start)
        if line_end == -1:
            line_end = n
        stripped_start = line_start
        while stripped_start < line_end and source[stripped_start] in ' \t':
            stripped_start += 1
        if source[stripped_start:stripped_start + 8] == '#include':
            rest = source[stripped_start + 8:line_end]
            q1 = rest.find('"')
            if q1 != -1:
                q2 = rest.find('"', q1 + 1)
                if q2 != -1:
                    abs_q1 = stripped_start + 8 + q1
                    abs_q2 = stripped_start + 8 + q2
                    for j in range(abs_q1, abs_q2 + 1):
                        in_include[j] = True
        line_start = line_end + 1

    def _count_trailing_backslashes(pos: int) -> int:
        count = 0
        j = pos - 1
        while j >= 0 and source[j] == '\\':
            count += 1
            j -= 1
        return count

    in_context = ''
    result: List[str] = []
    i = 0
    while i < n:
        c = source[i]

        if in_include[i] and not mask_includes and in_context == '':
            result.append(c)
            i += 1
            continue

        if in_context == '':
            if i in include_ends:
                end = include_ends[i]
                result.extend('x' * (end - i))
                i = end
                continue
            if c == '/' and i + 1 < n:
                if source[i + 1] == '/':
                    in_context = '//'
                    result.append('x')
                    i += 1
                    result.append('x')
                    i += 1
                    continue
                elif source[i + 1] == '*':
                    in_context = '/*'
                    result.append('x')
                    i += 1
                    result.append('x')
                    i += 1
                    continue
            if c == 'R' and i + 1 < n and source[i + 1] == '"':
                prefix_len = 0
                if i >= 2 and source[i-2:i] == 'u8':
                    boundary_pos = i - 3
                    if boundary_pos < 0 or not (source[boundary_pos].isalnum() or source[boundary_pos] == '_'):
                        prefix_len = 2
                elif i >= 1 and source[i-1] in 'uUL':
                    boundary_pos = i - 2
                    if boundary_pos < 0 or not (source[boundary_pos].isalnum() or source[boundary_pos] == '_'):
                        prefix_len = 1
                elif i == 0 or not (source[i-1].isalnum() or source[i-1] == '_'):
                    prefix_len = 0
                else:
                    prefix_len = -1

                if prefix_len >= 0:
                    if prefix_len > 0:
                        for j in range(prefix_len):
                            result[-(j + 1)] = 'x'

                    delim = ''
                    in_context = 'raw:'
                    result.append('x')
                    i += 1
                    result.append('x')
                    i += 1
                    paren_pos = source.find('(', i)
                    if paren_pos != -1:
                        delim = source[i:paren_pos]
                        for _ in range(i, paren_pos + 1):
                            result.append('x')
                        i = paren_pos + 1
                        in_context = 'raw:' + delim
                    continue
            if c == '"':
                in_context = '"'
                result.append('x')
                i += 1
                continue
            if c == "'":
                in_context = "'"
                result.append('x')
                i += 1
                continue
            result.append(c)
            i += 1

        elif in_context == '//':
            if c == '\n' or c == '\r':
                in_context = ''
                result.append(c)
            else:
                result.append('x')
            i += 1

        elif in_context == '/*':
            if c == '/' and i > 0 and source[i - 1] == '*':
                in_context = ''
            result.append('x')
            i += 1

        elif in_context == '"':
            if c == '\\':
                result.append('x' if mask_string_contents else c)
                i += 1
                if i < n:
                    result.append('x' if mask_string_contents else source[i])
                    i += 1
                continue
            if c == '"':
                in_context = ''
            result.append('x' if mask_string_contents else c)
            i += 1

        elif in_context == "'":
            if c == '\n' or c == '\r':
                # Unterminated char literal cannot cross a line: recover so
                # apostrophes (#error can't, 1'000) do not mask the file tail.
                in_context = ''
                result.append(c)
                i += 1
                continue
            if c == '\\':
                result.append('x')
                i += 1
                if i < n:
                    result.append('x')
                    i += 1
                continue
            if c == "'":
                in_context = ''
            result.append('x')
            i += 1

        elif in_context.startswith('raw:'):
            delim = in_context[4:]
            close_seq = ')' + delim + '"'
            close_pos = source.find(close_seq, i)
            if close_pos == -1:
                for j in range(i, n):
                    result.append('x' if mask_string_contents else source[j])
                i = n
            else:
                for j in range(i, close_pos):
                    result.append('x' if mask_string_contents else source[j])
                for _ in range(close_pos, close_pos + len(close_seq)):
                    result.append('x')
                i = close_pos + len(close_seq)
                in_context = ''
        else:
            result.append(c)
            i += 1

    return ''.join(result)

_BARE_ONLY_INCLUDES = {
    # mma.h handled via .ppu_compat/mma.h shim
    "cuda.h": "hggc.h",
    "nvToolsExt.h": "hgToolsExt.h",
    "nccl.h": "pccl.h",
}


_RE_LINE_SPLICE = re.compile(r"\\\r?\n")
_HEADER_SPACE = r"(?:[ \t\f\v]|/\*(?:[^*]|\*(?!/))*\*/)*"
_RE_HEADER_NAME = re.compile(
    rf"(?:^{_HEADER_SPACE}#{_HEADER_SPACE}include(?:_next)?{_HEADER_SPACE}"
    rf"|(?P<probe>(?<!\w)__has_include(?:_next)?){_HEADER_SPACE}\({_HEADER_SPACE})"
    r'(?P<header><[^>\r\n]+>|"[^"\r\n]+")'
    rf"(?(probe)(?={_HEADER_SPACE}\)))",
    re.MULTILINE,
)


def _iter_header_names(source: str):
    splice_positions = []
    splice_shifts = [0]

    def _remove_splice(match):
        splice_positions.append(match.start() - splice_shifts[-1])
        splice_shifts.append(splice_shifts[-1] + len(match.group()))
        return ""

    logical_source = _RE_LINE_SPLICE.sub(_remove_splice, source)
    probe_mask = None
    for match in _RE_HEADER_NAME.finditer(logical_source):
        if match.group("probe") is not None:
            if probe_mask is None:
                probe_mask = _mask_comments_and_strings(logical_source)
            if probe_mask[match.start("probe")] == 'x':
                continue
        start, end = match.span("header")
        # Include delimiters anchor the mask even when the filename starts with a splice.
        source_start = start + splice_shifts[bisect_right(splice_positions, start)]
        source_end = end + splice_shifts[bisect_right(splice_positions, end - 1)]
        yield source_start, source_end, match.group("header")[1:-1]


def _rewrite_include_directives(source: str, ppu_sdk_version=None) -> str:
    """Rewrite header paths in include directives and header probes."""
    from .cuda_to_ppu_mappings import CUDA_INCLUDE_MAP

    mapping = dict(CUDA_INCLUDE_MAP)
    versioned_includes = _get_versioned_mappings("include", ppu_sdk_version)
    mapping.update(versioned_includes)
    include_prefixes = _get_versioned_mappings("include_prefix", ppu_sdk_version)
    _sorted_pairs = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    masked = None
    parts = []
    last_end = 0
    for start, end, path in _iter_header_names(source):
        # Skip directives inside comments (comments mask every char, so
        # probe a path char that is not a literal 'x').  String and raw
        # string bodies stay rewritable, like the identifier passes.
        if masked is None:
            masked = _mask_comments_and_strings(source)
        probe = next(
            (j for j in range(start + 1, end - 1) if source[j] != 'x'),
            None,
        )
        if probe is not None and masked[probe] == 'x':
            continue
        new_path = path

        if '/' not in new_path and new_path in _BARE_ONLY_INCLUDES:
            new_path = _BARE_ONLY_INCLUDES[new_path]

        for cuda_path, ppu_path in _sorted_pairs:
            if cuda_path == ppu_path:
                continue

            if '/' in cuda_path:
                if cuda_path in new_path:
                    new_path = new_path.replace(cuda_path, ppu_path)
            else:
                last_slash = new_path.rfind('/')
                filename = new_path[last_slash + 1:] if last_slash >= 0 else new_path
                # Versioned root-header renames are root-level only.
                root_only = '/' not in cuda_path and cuda_path in versioned_includes
                if filename == cuda_path and not (root_only and last_slash >= 0):
                    if last_slash >= 0:
                        new_path = new_path[:last_slash + 1] + ppu_path
                    else:
                        new_path = ppu_path

        # Rewrite the leading path component only (thrust/ -> sail/thrust/).
        # A '.' after the stem would also drag in unrelated project
        # headers (cooperative_groups.config.h); bare cooperative_groups.h
        # stays covered via the exact ".h" case.
        for inc_prefix, inc_replacement in include_prefixes.items():
            if '/' in inc_prefix:
                if new_path.startswith(inc_prefix):
                    new_path = inc_replacement + new_path[len(inc_prefix):]
            else:
                stem_len = len(inc_prefix)
                rest = new_path[stem_len:]
                if (
                    new_path[:stem_len] == inc_prefix
                    and (rest == '' or rest[0] == '/' or rest == '.h')
                ):
                    new_path = inc_replacement + rest

        if new_path != path:
            parts.append(source[last_end:start])
            parts.append(source[start] + new_path + source[end - 1])
            last_end = end
    parts.append(source[last_end:])
    return "".join(parts)


# ── Preprocessor (per-file) ───────────────────────────────────────────

def _ppu_repl(match, identifier_map):
    """Regex replacement callback."""
    word = match.group(1)
    if word in _CUSTOM_SPECIAL_MAP:
        return _CUSTOM_SPECIAL_MAP[word]
    return identifier_map.get(word, _PPU_MAP.get(word, word))


def _masked_replace(source, regex, repl, mask_includes=False):
    """Replace *regex* matches in *source*, skipping comments, string
    literals and (optionally) include paths."""
    masked = _mask_comments_and_strings(source, mask_includes=mask_includes)
    parts = []
    last = 0
    for m in regex.finditer(source):
        if masked[m.start()] == "x":
            continue
        parts.append(source[last:m.start()])
        parts.append(repl(m))
        last = m.end()
    parts.append(source[last:])
    return "".join(parts)


_NS_RE_CACHE: Dict[tuple, re.Pattern] = {}

_NS_WS = " \t\r\n\x0b\x0c"
_NS_IDENT = re.compile(r"[A-Za-z_]\w*")
# Keywords after which a following ::name is still the global scope;
# any other identifier there means a nested qualification (foo ::name).
_NS_GLOBAL_KEYWORDS = {
    "using", "namespace", "typename", "template", "friend", "typedef",
    "return", "case", "new", "delete", "throw", "else", "do",
    "co_await", "co_return", "co_yield", "sizeof", "alignof", "typeid",
    "decltype", "noexcept", "operator",
    # Declaration specifiers and type keywords.
    "const", "constexpr", "static", "extern", "inline", "virtual",
    "explicit", "mutable", "volatile", "unsigned", "signed", "long",
    "short", "auto", "void", "bool", "char", "int", "float", "double",
    "class", "struct", "union", "enum", "public", "private", "protected",
}
# ')' makes ::name global after these, nested otherwise (decltype(x)::cub).
_NS_STMT_PAREN_KEYWORDS = {"if", "for", "while", "switch"}


def _namespace_regex(ns_map):
    """Regex matching bare namespace words (thrust::x, namespace thrust)
    or globally qualified ones (::thrust::x).  Lookbehinds keep sail::*
    and foo::cub untouched while still allowing case 1:thrust::x
    (single ':') and keyword::word (return::cub, namespace::thrust);
    '/' stays allowed: include paths are masked by this pass.  Real
    namespace references are decided in _namespace_repl_factory."""
    key = tuple(sorted(ns_map))
    regex = _NS_RE_CACHE.get(key)
    if regex is None:
        alt = "|".join(re.escape(k) for k in key)
        regex = re.compile(
            r"(?<![\w.])(?<!::)(" + alt + r")(?![\w])"
            r"|::(" + alt + r")(?![\w])"
        )
        _NS_RE_CACHE[key] = regex
    return regex


def _ns_next_token(source, pos):
    """First significant token at/after *pos* ('::' or one word/char)."""
    i, n = pos, len(source)
    while i < n:
        c = source[i]
        if c in _NS_WS:
            i += 1
        elif c == '/' and i + 1 < n and source[i + 1] == '/':
            j = source.find('\n', i)
            i = n if j == -1 else j + 1
        elif c == '/' and i + 1 < n and source[i + 1] == '*':
            j = source.find('*/', i + 2)
            i = n if j == -1 else j + 2
        else:
            break
    if i >= n:
        return None
    if source[i:i + 2] == '::':
        return '::'
    m = _NS_IDENT.match(source, i)
    return m.group(0) if m else source[i]


def _ns_prev_token(source, pos):
    """(token, start) of the last significant token before *pos*, or
    (None, 0) at start of file.  Skips whitespace and comments."""
    i = pos
    while i > 0:
        c = source[i - 1]
        if c in _NS_WS:
            i -= 1
        elif i >= 2 and source[i - 2:i] == '*/':
            j = source.rfind('/*', 0, i - 2)
            i = 0 if j == -1 else j
        else:
            # Line comments: only when the text before the word on this
            # line has an even number of quotes (so "a//b" is a string).
            line_start = source.rfind('\n', 0, i) + 1
            k = source.rfind('//', line_start, i)
            if k != -1 and source.count('"', line_start, k) % 2 == 0:
                i = k  # comment runs to end of line
                continue
            break
    if i <= 0:
        return None, 0
    j = i - 1
    if not (source[j].isalnum() or source[j] == '_'):
        return source[j], j
    while j >= 0 and (source[j].isalnum() or source[j] == '_'):
        j -= 1
    return source[j + 1:i], j + 1


def _ns_paren_owner(source, close_pos):
    """Token before the '(' matching the ')' at *close_pos*, or None."""
    depth = 0
    i = close_pos
    while i >= 0:
        c = source[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
            if depth == 0:
                tok, _ = _ns_prev_token(source, i)
                return tok
        i -= 1
    return None


def _ns_is_nested_before(source, prev, prev_start):
    """True when *prev* makes a following ::name a nested qualification."""
    if prev == '>':
        return not (prev_start > 0 and source[prev_start - 1] == '-')  # -> vs T>
    if prev == ')':
        return _ns_paren_owner(source, prev_start) not in _NS_STMT_PAREN_KEYWORDS
    if prev == ']':
        return True
    return bool(_NS_IDENT.fullmatch(prev or '')) and prev not in _NS_GLOBAL_KEYWORDS


def _namespace_repl_factory(ns_map, source):
    # Equal-length mask for paren-owner scans: brackets inside comments
    # and string literals must not affect nesting depth.
    scan_source = _mask_comments_and_strings(source, mask_string_contents=True)

    def _repl(match):
        if match.group(2) is not None:
            # ::word: keep only the global scope.  foo ::word / foo<T>::word
            # are nested qualifications (leading :: belongs to the left
            # operand), so require a non-identifier (or keyword) before ::.
            prev, prev_start = _ns_prev_token(source, match.start())
            if prev is not None and _ns_is_nested_before(scan_source, prev, prev_start):
                return match.group(0)
            # A bare namespace is not a value: rewrite only a
            # qualification or an alias / using-namespace RHS, not a
            # same-named global variable.
            word = match.group(2)
            nxt = _ns_next_token(source, match.end())
            if nxt == '::':
                return "::" + ns_map[word]
            if prev == 'namespace' and nxt == ';':
                return "::" + ns_map[word]      # using namespace ::cub;
            if prev == '=':
                name, name_start = _ns_prev_token(source, prev_start)
                kw, _ = (
                    _ns_prev_token(source, name_start) if name else (None, 0)
                )
                if name and _NS_IDENT.fullmatch(name) and kw == 'namespace':
                    return "::" + ns_map[word]  # namespace NAME = ::word;
            return match.group(0)
        # Bare word: rewrite only real namespace usages so variable names
        # (int thrust), alias declarators (namespace cub = hipcub) and
        # namespace definitions survive, while thrust::x, using directives
        # and alias right-hand sides still migrate.
        word = match.group(1)
        nxt = _ns_next_token(source, match.end())
        prev, prev_start = _ns_prev_token(source, match.start())
        if (
            prev == ':'
            and prev_start > 0
            and source[prev_start - 1] == ':'
        ):
            # word sits right of a (possibly spaced) '::'
            before, before_start = _ns_prev_token(source, prev_start - 1)
            if nxt == '::':
                # nested when qualified from the left (foo:: word,
                # sail:: thrust), global when the '::' opens the scope.
                if before is not None and _ns_is_nested_before(
                        scan_source, before, before_start):
                    return match.group(0)
                return ns_map[word]
            # `namespace NAME = :: word;` and `using namespace :: word;`
            # resolve through the scope operator.
            if before in ('=', 'namespace'):
                prev, prev_start = before, before_start
        if nxt == '::':
            return ns_map[word]
        if prev == 'namespace' and nxt == ';':
            return ns_map[word]      # using namespace cub;
        if prev == '=':
            # namespace NAME = word; (alias right-hand side)
            name, name_start = _ns_prev_token(source, prev_start)
            kw, _ = _ns_prev_token(source, name_start) if name else (None, 0)
            if name and _NS_IDENT.fullmatch(name) and kw == 'namespace':
                return ns_map[word]
        return match.group(0)
    return _repl


def _rewrite_namespaces(source, ppu_sdk_version):
    """Apply the SDK-versioned namespace migrations (thrust -> sail::thrust)."""
    ns_map = _get_versioned_mappings("namespace", ppu_sdk_version)
    if not ns_map:
        return source
    return _masked_replace(
        source, _namespace_regex(ns_map), _namespace_repl_factory(ns_map, source),
        mask_includes=True,
    )


def preprocessor(
    output_directory: str,
    filepath: str,
    all_files: Iterable[str],
    stats: Dict[str, list],
    clean_ctx: Optional[object],
    show_progress: bool,
    project_directory: str = "",
    backup: bool = False,
    inject_compat_include: bool = False,
    inject_compat_include_ignores: Iterable[str] = (),
    compat_include_line: str = COMPAT_WRAPPER_INCLUDE_LINE,
    ppu_sdk_version=None,
) -> SailifyResult:
    """Process a single CUDA source file and write the PPU equivalent.

    An omitted *ppu_sdk_version* disables version-conditional mappings.

    With *inject_compat_include*, header files (.h/.hpp/.cuh) additionally get
    the compat wrapper include (*compat_include_line*) prepended.

    *inject_compat_include_ignores* — glob patterns (relative to
    *project_directory*) of headers to exempt from the injection.
    """
    if project_directory:
        rel = os.path.relpath(filepath, project_directory)
    else:
        rel = os.path.relpath(filepath, os.path.dirname(filepath))
    ppu_rel = get_ppu_file_path(rel)
    if output_directory:
        out_path = os.path.join(output_directory, ppu_rel)
    else:
        out_path = os.path.join(os.path.dirname(filepath), ppu_rel)

    result = SailifyResult(CurrentState.INITIALIZED, os.path.abspath(out_path))

    try:
        with open(filepath, "r", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        result.status = f"read error: {exc}"
        return result

    output_source = source

    # 1. Apply custom mappings
    if _CUSTOM_SPECIAL_MAP:
        _re_custom = re.compile(
            r"(?<!\w)(" + _CUSTOM_TRIE.export_to_regex() + r")(?!\w)"
        )
        output_source = _re_custom.sub(
            lambda m: _CUSTOM_SPECIAL_MAP.get(m.group(1), m.group(1)),
            output_source,
        )

    # 1b. Apply include-specific path rewrites BEFORE the trie step
    output_source = _rewrite_include_directives(output_source, ppu_sdk_version)

    # 2. Apply main CUDA→PPU mappings via the pre-compiled trie regex
    identifier_map = _get_versioned_mappings("identifier", ppu_sdk_version)
    _masked = _mask_comments_and_strings(output_source)
    _masked_identifiers = None

    _new_parts = []
    _last_end = 0
    for _m in _RE_PPU_PREPROCESSOR.finditer(output_source):
        if _m.group(1) in identifier_map:
            if _masked_identifiers is None:
                _masked_identifiers = _mask_comments_and_strings(output_source, mask_includes=True)
            if _masked_identifiers[_m.start()] == 'x':
                continue
        elif _masked[_m.start()] == 'x':
            continue
        _new_parts.append(output_source[_last_end:_m.start()])
        _new_parts.append(_ppu_repl(_m, identifier_map))
        _last_end = _m.end()
    _new_parts.append(output_source[_last_end:])
    output_source = "".join(_new_parts)

    # 2b. SDK-versioned identifier-prefix families (SANITIZER_CBID_*).
    # Single alternation, longest prefix first so shorter ones cannot
    # shadow longer ones; includes are masked (sailify renames headers
    # itself, never project files).
    identifier_prefixes = _get_versioned_mappings(
        "identifier_prefix", ppu_sdk_version)
    if identifier_prefixes:
        _sorted_prefixes = sorted(identifier_prefixes, key=len, reverse=True)
        _prefix_regex = re.compile(
            r"(?<!\w)("
            + "|".join(re.escape(_p) for _p in _sorted_prefixes)
            # \w keeps whole-token boundaries; also allow token pasting
            # (SANITIZER_CBID_##domain, spaces allowed around ##).
            + r")(?=\w|[ \t]*##)")
        output_source = _masked_replace(
            output_source,
            _prefix_regex,
            lambda m: identifier_prefixes[m.group(1)],
            mask_includes=True,
        )

    # 2c. SDK-versioned namespace migrations (thrust -> sail::thrust).
    output_source = _rewrite_namespaces(output_source, ppu_sdk_version)

    for _pat, _repl in _EXTRA_MAPPING_LISTS:
        output_source = output_source.replace(_pat, _repl)

    if (
        inject_compat_include
        and os.path.splitext(filepath)[1] in DEFAULT_HEADER_EXTENSIONS
        and not any(_matches_pattern(rel.replace(os.sep, "/"), pat)
                    for pat in inject_compat_include_ignores)
    ):
        output_source = _inject_compat_include(output_source, compat_include_line)

    # Write output
    try:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        is_inplace = not output_directory or os.path.abspath(output_directory) == os.path.abspath(project_directory)
        if backup and is_inplace and os.path.isfile(filepath):
            backup_path = filepath + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(filepath, backup_path)
                if show_progress:
                    log.info("  Backup: %s → %s", filepath, backup_path)

        with open(out_path, "w") as fh:
            fh.write(output_source)
        if show_progress:
            log.info("  %s → %s", filepath, out_path)
    except OSError as exc:
        result.status = f"write error: {exc}"
        return result

    result.current_state = CurrentState.DONE
    result.status = "ok"
    stats.setdefault("ppuified", []).append(filepath)
    return result


# ── Directory operations ─────────────────────────────────────────────

def _copytree_compat(src, dst, ignore=None):
    """Copy directory tree from *src* to *dst*, merging if *dst* exists."""
    if not os.path.isdir(dst):
        shutil.copytree(
            src, dst,
            ignore=ignore or shutil.ignore_patterns(".git", "__pycache__", "build"),
            symlinks=True,
        )
        return

    ignore_fn = ignore or shutil.ignore_patterns(".git", "__pycache__", "build")
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in ignore_fn(dirpath, dirnames)]
        reldir = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst, reldir)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in filenames:
            if fname in ignore_fn(dirpath, filenames):
                continue
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(dst_dir, fname)
            try:
                if os.path.islink(src_file):
                    link_target = os.readlink(src_file)
                    if os.path.exists(dst_file) or os.path.islink(dst_file):
                        os.remove(dst_file)
                    os.symlink(link_target, dst_file)
                else:
                    shutil.copy2(src_file, dst_file)
            except OSError:
                pass


# ── Main entry point ─────────────────────────────────────────────────

def _print_detected_versions(ppu_sdk_version, runtime_api_version):
    if ppu_sdk_version:
        print(f"sailify: PPU SDK {format_version_string(ppu_sdk_version)} detected "
              f"(PPU_SDK_VERSION {version_to_int(ppu_sdk_version)})")
    else:
        print("sailify: PPU SDK version unknown; keeping legacy fixups "
              "(set $PPU_SDK or pass --ppu-sdk-version)")
    cuda_default = "12.9" if runtime_api_version == 2 else "13.0"
    print(f"sailify: hggcrt Runtime API v{runtime_api_version} "
          f"(COMPATIBLE_* defaults follow CUDA {cuda_default})")


def sailify(
    project_directory: str,
    output_directory: str = "",
    excludes: list = None,
    extensions: tuple = DEFAULT_EXTENSIONS,
    includes: list = None,
    ignores: list = None,
    custom_map_list: str = "",
    version_config: dict = None,
    backup: bool = False,
    install_ppu_compat: bool = True,
    ppu_compat_dir: str = "",
    inject_compat_include: bool = False,
    inject_compat_include_ignores: list = None,
    extra_mapping: str = "",
    verbose: bool = False,
    ppu_sdk_version: tuple = None,
    stub_probe: bool = True,
) -> dict:
    """Convert CUDA code to PPU code in the given project directory.

    *ppu_sdk_version* — explicit (major, minor[, patch]) PPU SDK tuple for
    the version-conditional fixups.  When None it is auto-detected from the
    environment (see ppu_sdk_version.detect_ppu_sdk_version).

    *stub_probe* — when True (default), stub only APIs the installed SDK
    does not export (see unsupported_stubs.probe); when False, emit every
    unsupported_apis.json entry with its version gate.

    *inject_compat_include* — prepend the compat wrapper include to every
    converted header (.h/.hpp/.cuh).  The path segment follows the name of
    *ppu_compat_dir* (default ``.ppu_compat``); consumers must put the PARENT
    of the compat dir on their include search path.

    *inject_compat_include_ignores* — glob patterns (relative to
    *project_directory*) of headers exempt from the injection.
    """
    if not os.path.isdir(project_directory):
        raise FileNotFoundError(f"Project directory not found: {project_directory}")

    _added_skip_dirs = []
    _rel_excludes = []
    if excludes:
        for exc in excludes:
            normalized = exc.replace("\\", "/").strip("/")
            if "/" in normalized:
                # Multi-level path: use relative path matching
                _rel_excludes.append(normalized)
            else:
                # Basename only: add to SKIP_DIRS
                if exc not in SKIP_DIRS:
                    SKIP_DIRS.add(exc)
                    _added_skip_dirs.append(exc)

    try:
        update_custom_mappings(custom_map_list or "")
        load_extra_mappings(extra_mapping or "")

        project_directory = os.path.abspath(project_directory)
        if output_directory:
            output_directory = os.path.abspath(output_directory)
            _out_of_place = os.path.abspath(output_directory) != project_directory
            if _out_of_place:
                _copytree_compat(project_directory, output_directory)
        else:
            output_directory = project_directory

        compat_dir = ppu_compat_dir or os.path.join(output_directory, ".ppu_compat")
        include_line = compat_wrapper_include_line(compat_dir)
        _compat_dir_name = os.path.basename(os.path.normpath(compat_dir))
        if _compat_dir_name not in SKIP_DIRS:
            SKIP_DIRS.add(_compat_dir_name)
            _added_skip_dirs.append(_compat_dir_name)

        _includes = includes or ["*"]
        _ignores = ignores or []

        all_files = list(matched_files_iter(
            project_directory, includes=_includes, ignores=_ignores,
            extensions=extensions, excludes=_rel_excludes,
        ))

        stats = {}
        if verbose:
            print(f"sailify: processing {len(all_files)} files from {project_directory}")

        if version_config is None:
            version_config = {}

        if ppu_sdk_version is None:
            ppu_sdk_version = detect_ppu_sdk_version()
        if install_ppu_compat:
            runtime_api_version = detect_runtime_api_version()
            _print_detected_versions(ppu_sdk_version, runtime_api_version)
            write_compat_wrapper_header(
                compat_dir,
                version_config if version_config else None,
                ppu_sdk_version=ppu_sdk_version,
                runtime_api_version=runtime_api_version,
                stub_probe=stub_probe,
            )
        elif inject_compat_include:
            log.warning(
                "inject_compat_include is on but install_ppu_compat is off: converted "
                "headers will reference %s which is not generated in this run.",
                include_line,
            )

        results = {}

        for filepath in all_files:
            res = preprocessor(
                output_directory=output_directory,
                filepath=filepath,
                all_files=all_files,
                stats=stats,
                clean_ctx=None,
                show_progress=verbose,
                project_directory=project_directory,
                backup=backup,
                inject_compat_include=inject_compat_include,
                inject_compat_include_ignores=inject_compat_include_ignores or [],
                compat_include_line=include_line,
                ppu_sdk_version=ppu_sdk_version,
            )
            results[os.path.abspath(filepath)] = res

        if verbose:
            n_ok = sum(1 for r in results.values() if r.status == "ok")
            print(f"sailify: done. {n_ok}/{len(results)} source files converted.")

    finally:
        for d in _added_skip_dirs:
            SKIP_DIRS.discard(d)

    return results


def _ensure_ppu_compat(output_directory: str, stub_probe: bool = True) -> None:
    """(Re)generate .ppu_compat in output_directory with the detected PPU SDK version."""
    compat_dir = os.path.join(output_directory, ".ppu_compat")
    ppu_sdk_version = detect_ppu_sdk_version()
    runtime_api_version = detect_runtime_api_version()
    _print_detected_versions(ppu_sdk_version, runtime_api_version)
    write_compat_wrapper_header(compat_dir, ppu_sdk_version=ppu_sdk_version,
                                runtime_api_version=runtime_api_version,
                                stub_probe=stub_probe)


def sailify_extra_files(
    output_directory: str,
    extra_files: list,
    show_detailed: bool = False,
    install_ppu_compat: bool = True,
    inject_compat_include: bool = False,
    inject_compat_include_ignores: list = None,
    stub_probe: bool = True,
) -> dict:
    """Sailify only the specified extra_files.

    Converts a list of user-provided CUDA source files in-place and returns
    a dict mapping absolute input path -> SailifyResult.
    """
    output_directory = os.path.abspath(output_directory)
    ppu_sdk_version = detect_ppu_sdk_version()
    if install_ppu_compat:
        _ensure_ppu_compat(output_directory, stub_probe=stub_probe)
    stats: Dict[str, list] = {}
    results = {}

    for filepath in extra_files:
        filepath = os.path.abspath(filepath)
        res = preprocessor(
            output_directory=output_directory,
            filepath=filepath,
            all_files=extra_files,
            stats=stats,
            clean_ctx=None,
            show_progress=show_detailed,
            project_directory=output_directory,
            inject_compat_include=inject_compat_include,
            inject_compat_include_ignores=inject_compat_include_ignores or [],
            ppu_sdk_version=ppu_sdk_version,
        )
        results[filepath] = res

    return results


# ── Recursive dependency discovery ───────────────────────────────────
_RE_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]+[<"]([^>"]+)[>"]',
    re.MULTILINE,
)


def _discover_includes(
    filepath: str,
    header_include_dirs: List[str],
    visited: set,
    discovered: List[str],
) -> None:
    """Parse #include directives from *filepath* and enqueue discovered deps.

    For each #include, tries to resolve the file against:
      1. The including file's own directory
      2. Each directory in *header_include_dirs*

    Only files with extensions (.cu, .cuh, .h, .hpp, etc.) that
    exist on disk and haven't been visited yet are added to *discovered*.
    """
    try:
        with open(filepath, "r", errors="replace") as fh:
            source = fh.read()
    except OSError:
        return

    file_dir = os.path.dirname(filepath)

    for m in _RE_INCLUDE.finditer(source):
        inc_path = m.group(1)

        candidates = []
        candidates.append(os.path.join(file_dir, inc_path))
        for inc_dir in header_include_dirs:
            candidates.append(os.path.join(inc_dir, inc_path))

        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in visited:
                break
            if os.path.isfile(candidate):
                if ".ppu_compat" in candidate.split(os.sep):
                    break
                ext = os.path.splitext(candidate)[1]
                if ext in DEFAULT_EXTENSIONS:
                    visited.add(candidate)
                    discovered.append(candidate)
                break


def sailify_extra_files_recursive(
    output_directory: str,
    extra_files: list,
    header_include_dirs: list = None,
    show_detailed: bool = False,
    backup: bool = False,
    install_ppu_compat: bool = True,
    inject_compat_include: bool = False,
    inject_compat_include_ignores: list = None,
    stub_probe: bool = True,
) -> dict:
    """Sailify extra_files with recursive #include dependency discovery.

    Like sailify_extra_files, but also discovers and processes #include'd
    header files recursively.  For each file being processed, parses its
    #include directives, resolves them against header_include_dirs and
    the file's own directory, and processes any source files found.

    Args:
        output_directory: Output directory for converted files.
        extra_files: List of user-provided CUDA source files to convert.
        header_include_dirs: Directories to search for #include'd headers.
        show_detailed: If True, log each file conversion.
        backup: If True, create .bak backups of files processed in-place.
        install_ppu_compat: If True, (re)generate .ppu_compat in output_directory.
        inject_compat_include: If True, prepend the .ppu_compat wrapper
            include to converted headers (see sailify()).
        inject_compat_include_ignores: Glob patterns of headers exempt from
            the injection (see sailify()).

    Returns:
        Dict mapping absolute input path -> SailifyResult.
    """
    output_directory = os.path.abspath(output_directory)
    ppu_sdk_version = detect_ppu_sdk_version()
    if install_ppu_compat:
        _ensure_ppu_compat(output_directory, stub_probe=stub_probe)
    _header_include_dirs = header_include_dirs or []
    stats: Dict[str, list] = {}
    results = {}

    # BFS queue: start with user-provided files
    visited: set = set()
    queue: List[str] = []

    for f in extra_files:
        f = os.path.abspath(f)
        if f not in visited:
            visited.add(f)
            queue.append(f)

    processed_count = 0
    discovered_count = 0
    initial_count = len(queue)

    while queue:
        filepath = queue.pop(0)

        # Discover #include dependencies before processing
        _discover_includes(filepath, _header_include_dirs, visited, queue)

        res = preprocessor(
            output_directory=output_directory,
            filepath=filepath,
            all_files=list(visited),
            stats=stats,
            clean_ctx=None,
            show_progress=show_detailed,
            project_directory=output_directory,
            backup=backup,
            inject_compat_include=inject_compat_include,
            inject_compat_include_ignores=inject_compat_include_ignores or [],
            ppu_sdk_version=ppu_sdk_version,
        )
        results[filepath] = res
        processed_count += 1

    discovered_count = processed_count - initial_count
    if show_detailed and discovered_count > 0:
        log.info(
            "Recursive sailify: processed %d files (%d from extra_files, %d discovered)",
            processed_count, initial_count, discovered_count,
        )

    return results

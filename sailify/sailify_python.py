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
build scripts.  The one exception is the opt-in derived-tree mode, which
writes converted files under caller-renamed directory components.
"""

import glob
import os
import re
import json
import shutil
import logging
import fnmatch
from enum import Enum
from typing import Iterable, Iterator, Optional, Dict, List, Tuple

from .cuda_to_ppu_mappings import (
    CUDA_TO_PPU_MAPPINGS,
    PPU_SDK_2_2_IDENTIFIER_MAP,
)
from .unsupported_stubs import write_stubs_header
from .ppu_sdk_version import (
    detect_ppu_sdk_version,
    detect_runtime_api_version,
    format_version_string,
    ppu_sdk_include_dirs,
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


def write_compat_wrapper_header(compat_dir, version_config=None,
                                ppu_sdk_version=None, runtime_api_version=None):
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

    write_stubs_header(compat_dir)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for hdr in _COMPAT_STATIC_HEADERS:
        src = os.path.join(pkg_dir, hdr)
        if os.path.isfile(src):
            dst = os.path.join(compat_dir, hdr)
            shutil.copy2(src, dst)
            log.info("Copied %s to %s", hdr, compat_dir)


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

    def __init__(self, current_state: CurrentState = None, ppuified_path: str = ""):
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

_RE_PPU_PREPROCESSOR = re.compile(
    r"(?<!\w)(" + _PPU_TRIE.export_to_regex() + r")(?!\w)"
)

# ── SDK-version-conditional mappings (PPU SDK >= 2.2) ─────────────────

_SDK_2_2_VERSION_INT = 20200  # version_to_int((2, 2, 0))

_SDK22_MAP: Dict[str, str] = {}
_SDK22_RE: Optional[re.Pattern] = None
# (include-dirs key, frozenset of identifiers the SDK declares): keyed by the
# include roots so swapping the SDK env (or a test faking one) re-probes.
_SDK22_PROBED: Optional[Tuple[Tuple[str, ...], frozenset]] = None


def _probe_sdk22_identifiers(targets) -> frozenset:
    """Which of `targets` the active PPU SDK actually declares.

    The 2.2-era identifier map rewrites CUDA spellings to __hggc_ ones, but
    whether an SDK ships the prefixed or the unprefixed spelling is a layout
    property, not a version property: wt_v3's 2.2 SDK declares the
    __hggc_ bf16/pipeline intrinsics in regular TUs, while other 2.3-era
    SDKs declare only the unprefixed CUDA names there.  Rewriting towards a
    spelling the SDK does not provide would produce undeclared-identifier
    errors (exactly what the flat-layout SDKs show), so each pair activates
    only when its target identifier appears in the SDK's headers.  Probe
    result is cached per include-dirs tuple until the map is reset.
    """
    global _SDK22_PROBED
    dirs = tuple(ppu_sdk_include_dirs())
    if _SDK22_PROBED is not None and _SDK22_PROBED[0] == dirs:
        return _SDK22_PROBED[1]
    found = set()
    remaining = set(targets)
    if remaining:
        blob_re = re.compile(
            b"(?<!\\w)(" + b"|".join(sorted(re.escape(t.encode()) for t in remaining))
            + b")(?!\\w)"
        )
        for include_dir in dirs:
            for dirpath, _dirnames, filenames in os.walk(include_dir):
                for name in filenames:
                    if not name.endswith((".h", ".hpp")):
                        continue
                    try:
                        with open(os.path.join(dirpath, name), "rb") as fh:
                            blob = fh.read()
                    except OSError:
                        continue
                    for m in blob_re.finditer(blob):
                        found.add(m.group(0).decode())
                    if targets <= found:
                        break
                if targets <= found:
                    break
    _SDK22_PROBED = (dirs, frozenset(found))
    return _SDK22_PROBED[1]


def _enable_sdk_2_2_mappings() -> None:
    """Activate the PPU SDK >= 2.2 identifier group (idempotent).

    Pairs whose __hggc_/__HG_ target the active SDK does not declare stay
    off (the CUDA spelling keeps compiling there)."""
    global _SDK22_MAP, _SDK22_RE
    available = _probe_sdk22_identifiers(set(PPU_SDK_2_2_IDENTIFIER_MAP.values()))
    trie = Trie()
    mapping = {}
    skipped = []
    for src, dst in PPU_SDK_2_2_IDENTIFIER_MAP.items():
        if dst in available:
            trie.add(src)
            mapping[src] = dst
        else:
            skipped.append(src)
    if skipped:
        log.info(
            "PPU SDK does not declare %d of the 2.2-era target spellings; "
            "keeping the CUDA names for: %s",
            len(skipped), ", ".join(sorted(skipped)[:8]) + (", ..." if len(skipped) > 8 else ""),
        )
    _SDK22_MAP = mapping
    _SDK22_RE = (
        re.compile(r"(?<!\w)(" + trie.export_to_regex() + r")(?!\w)")
        if mapping else None
    )
    _enable_sail_include_prefixes()


def _maybe_enable_sdk_2_2_mappings(ppu_sdk_version) -> None:
    global _SDK22_MAP, _SDK22_RE, _SDK22_PROBED
    if ppu_sdk_version is not None and version_to_int(ppu_sdk_version) >= _SDK_2_2_VERSION_INT:
        _enable_sdk_2_2_mappings()
    else:
        _SDK22_MAP = {}
        _SDK22_RE = None
        _SDK22_PROBED = None


# ── sail-nested cccl layout (wt_v3-era PPU SDKs) ──────────────────────

# wt_v3-era SDKs nest the cccl headers (thrust/cub/cooperative_groups, and
# the hggc libcu++ port once the SDK ships it there) under
# <SDK>/include/sail/ instead of the include root.  The include rewriter
# prefixes those families with "sail/" so converted sources need no
# build-system -I workaround.  Families activate only when the SDK actually
# provides <sail>/<family>; #include <sail/...> paths are left untouched.
# The SDK compiles those headers inside `namespace sail` (its
# wrapped-namespace default), so the namespace rewriter below also
# prefixes the identifiers themselves — converted code speaks
# sail::thrust / sail::cub / sail::cooperative_groups directly instead of
# relying on global-namespace aliases in ppu_sdk_fixups.h.
_SAIL_INCLUDE_FAMILIES: tuple = ()
_CANDIDATE_SAIL_FAMILIES = ("thrust", "cub", "cooperative_groups", "hggc")
_SAIL_NS_FAMILIES = ("thrust", "cub", "cooperative_groups")
_SAIL_NS_COMBINED_RE = None

# CUDA core headers that wt_v3-era PPU SDKs ship only with the __hggc_
# spelling (hggc_vector_types.h, ...), while older/other layouts provide the
# unprefixed CUDA names.  Activated per pair only when the SDK include tree
# has the prefixed file but not the unprefixed one, so flat-layout SDKs
# (which provide the CUDA names natively and may lack the hggc_ ones) are
# never rewritten.
_PREFIXED_INCLUDE_MAP: Dict[str, str] = {}


def _enable_prefixed_core_includes() -> None:
    """Probe the SDK for CUDA core headers available only as hggc_*."""
    global _PREFIXED_INCLUDE_MAP
    if _PREFIXED_INCLUDE_MAP:
        return
    from .cuda_to_ppu_mappings import PPU_SDK_PREFIXED_CORE_HEADERS

    mapping = {}
    for bare, prefixed in PPU_SDK_PREFIXED_CORE_HEADERS.items():
        for include_dir in ppu_sdk_include_dirs():
            has_prefixed = os.path.isfile(os.path.join(include_dir, prefixed))
            has_bare = os.path.isfile(os.path.join(include_dir, bare))
            if has_prefixed:
                if not has_bare:
                    mapping[bare] = prefixed
                break
    _PREFIXED_INCLUDE_MAP = mapping
    if mapping:
        log.info(
            "PPU SDK ships CUDA core headers only as hggc_*: %s",
            ", ".join(mapping),
        )


def _enable_sail_include_prefixes() -> None:
    """Probe the SDK for <sail>/<family> headers and activate prefixes."""
    global _SAIL_INCLUDE_FAMILIES
    if _SAIL_INCLUDE_FAMILIES:
        return
    families = []
    for include_dir in ppu_sdk_include_dirs():
        for fam in _CANDIDATE_SAIL_FAMILIES:
            if fam in families:
                continue
            sail_dir = os.path.join(include_dir, "sail")
            if os.path.isdir(os.path.join(sail_dir, fam)) or os.path.isfile(
                os.path.join(sail_dir, fam + ".h")
            ):
                families.append(fam)
    _SAIL_INCLUDE_FAMILIES = tuple(families)
    if _SAIL_INCLUDE_FAMILIES:
        log.info(
            "PPU SDK nests headers under sail/: #include paths of %s get "
            "a sail/ prefix",
            ", ".join(_SAIL_INCLUDE_FAMILIES),
        )
    _enable_prefixed_core_includes()
    _enable_sail_namespace_rewrites()


def _enable_sail_namespace_rewrites() -> None:
    """Namespace-resolve the cccl families the SDK nests under sail/.

    Activated only for families whose headers live under <sail>/ (flat-layout
    SDKs keep the global namespaces and must not be rewritten).  The rewrite
    is context-scoped — plain identifiers named thrust/cub stay untouched
    (actlize uses `cub` as an integer loop variable) — covering only ROOT
    references:

      ::cub::x / cub::x            root-qualified uses
      ::cub                        root uses in macro arguments etc.
      namespace cub ...            namespace definitions, forwards, aliases,
                                   `using namespace cub`
      namespace cg = cooperative_groups;
                                   alias declarations of the whole namespace

    Nested qualifications (cuda::cub, at_cuda_detail::cub, foo::thrust) are
    left alone: the family is a member of the enclosing namespace there, and
    whether that enclosing namespace is a CUB_WRAPPED_NAMESPACE-style project
    namespace or a to-be-renamed persona namespace is project knowledge the
    token-level rewrite cannot have.  Redirecting them would guess wrong in
    at least the wrapped-namespace case (W::cub is already the correct
    reference; rewriting it produces W::sail::cub).  Such pairs belong in the
    project's extra_mapping.  Note `ROCM_HIPCUB(::cub)` still matches: the
    `::cub` there is preceded by `(`, not an identifier/`::`, so it is a root
    reference and does get rewritten.

    #include paths, comments and string literals are masked out by the
    caller's masked sweep.  The rewrite is idempotent (a reference already
    prefixed with sail:: is preceded by : or a word char and so never
    matches), so the conversion can re-run on already-staged sources.

    Rewrites are emitted with a LEADING :: (::sail::cub::x, not
    sail::cub::x): converted torch hosts its own sail persona namespaces
    (c10::sail, at::sail), and an unqualified sail:: inside namespace c10
    would wrongly resolve to c10::sail.  A real namespace-definition cannot
    start with :: (C++ grammar), so that form alone keeps the unqualified
    spelling — valid at global scope, where such definitions live.
    """
    global _SAIL_NS_COMBINED_RE
    if _SAIL_NS_COMBINED_RE is not None:
        return
    fams = [f for f in _SAIL_NS_FAMILIES if f in _SAIL_INCLUDE_FAMILIES]
    if not fams:
        return
    fam = "|".join(re.escape(f) for f in fams)
    _SAIL_NS_COMBINED_RE = re.compile(
        r"(?<![\w:])(?P<q1>%(fam)s)::"
        r"|(?<![\w:])::(?P<q2>%(fam)s)(?!\w)"
        r"|(?P<using>using\s+)?namespace\s+(?P<decl>%(fam)s)\b(?!\s*=)"
        r"|namespace\s+(?P<alias>\w+)\s*=\s*(?P<rhs>%(fam)s)\s*;"
        % {"fam": fam}
    )
    log.info(
        "Converted sources reference %s directly (source-level "
        "namespace resolution)",
        ", ".join("::sail::" + fam for fam in fams),
    )


def _sail_ns_repl(match) -> str:
    if match.group("q1") is not None:
        return "::sail::" + match.group("q1") + "::"
    if match.group("q2") is not None:
        return "::sail::" + match.group("q2")
    if match.group("decl") is not None:
        if match.group("using") is not None:
            return "using namespace ::sail::" + match.group("decl")
        return "namespace sail::" + match.group("decl")
    return "namespace %s = ::sail::%s;" % (
        match.group("alias"), match.group("rhs"))


def _sub_masked(pattern, repl, source: str, mask: str) -> str:
    """re.sub, but only where the pre-computed mask is not 'x'."""
    parts: List[str] = []
    last = 0
    for m in pattern.finditer(source):
        if mask[m.start()] == 'x':
            continue
        parts.append(source[last:m.start()])
        parts.append(repl(m))
        last = m.end()
    if not parts:
        return source
    parts.append(source[last:])
    return "".join(parts)


# Post-trie repairs for wrapper idioms that compose the wrapped-namespace
# name out of macro pieces (torch's cub wrapper, -DCUB_WRAPPED_NAMESPACE=
# at_cuda_detail) live in the project's extra-mapping file, not here:
# they are project-specific string replacements applied after this step.


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
# trie replacement step.  Loaded from a JSON file via --extra-mapping,
# either as a flat JSON array of [from, to] pairs (applied to every
# converted file) or as an object:
#
#     {"mappings": [[from, to], ...],          # every converted file
#      "derived_mappings": [[from, to], ...]}  # only derived-tree files
_EXTRA_MAPPING_LISTS = []
_DERIVED_EXTRA_MAPPING_LISTS = []


def load_extra_mappings(extra_mapping_path: str) -> None:
    """Load extra string-to-string mappings from a JSON file.
    Each pair is applied via ``str.replace`` on the post-trie source text.
    """
    global _EXTRA_MAPPING_LISTS, _DERIVED_EXTRA_MAPPING_LISTS
    if not extra_mapping_path or not os.path.isfile(extra_mapping_path):
        return
    try:
        with open(extra_mapping_path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load extra mappings: {e}")
        return
    if isinstance(data, dict):
        mappings = data.get("mappings", [])
        derived_mappings = data.get("derived_mappings", [])
        if not isinstance(mappings, list) or not isinstance(derived_mappings, list):
            log.warning(
                "Extra mapping object must hold lists under 'mappings' / "
                "'derived_mappings', ignoring")
            return
    elif isinstance(data, list):
        mappings, derived_mappings = data, []
    else:
        log.warning(
            "Extra mapping file must contain a JSON array or an object with "
            "'mappings'/'derived_mappings', ignoring")
        return
    _EXTRA_MAPPING_LISTS = [(str(src), str(dst)) for src, dst in mappings]
    _DERIVED_EXTRA_MAPPING_LISTS = [
        (str(src), str(dst)) for src, dst in derived_mappings]
    log.info(
        f"Loaded {len(_EXTRA_MAPPING_LISTS)} extra mappings "
        f"({len(_DERIVED_EXTRA_MAPPING_LISTS)} derived-only) "
        f"from {extra_mapping_path}")


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

# Derived-tree mode: mirror caller-chosen directory components into a
# derived tree.  Files whose path contains a renamed component are written
# to the mirrored path (original paths keep their pristine copies); "copy"
# glob patterns are mirrored byte-for-byte without conversion.  See the
# sailify() docstring for the config format.


def _normalize_derived_tree(derived_tree) -> Optional[dict]:
    """Validate/normalize the derived_tree config; None when disabled."""
    if not derived_tree:
        return None
    if not isinstance(derived_tree, dict) or "rename" not in derived_tree:
        raise ValueError(
            "derived_tree must be a dict like "
            '{"rename": {"<src-dir>": "<dst-dir>"}, "file_roots": [...], '
            '"include_roots": [...], "copy": [...]}'
        )
    if not derived_tree["rename"]:
        return None
    roots = derived_tree.get("file_roots")
    if roots is not None:
        if (not isinstance(roots, (list, tuple))
                or not all(isinstance(r, str) and r for r in roots)):
            raise ValueError(
                'derived_tree "file_roots" must be a list of non-empty '
                f"strings, got {roots!r}"
            )
        derived_tree = dict(derived_tree, file_roots=list(roots))
    copies = derived_tree.get("copy")
    if copies is not None:
        if (not isinstance(copies, (list, tuple))
                or not all(isinstance(c, str) and c for c in copies)):
            raise ValueError(
                'derived_tree "copy" must be a list of non-empty glob '
                f"patterns, got {copies!r}"
            )
        derived_tree = dict(derived_tree, copy=list(copies))
    return derived_tree


def _strip_root(path: str, roots) -> bool:
    if not roots:
        return True
    return path.startswith(tuple(r.replace("\\", "/").strip("/") + "/" for r in roots))


def _derive_tree_segments(path: str, roots, rename: dict) -> str:
    """Apply *rename* to directory components of *path* under *roots*."""
    if not _strip_root(path, roots):
        return path
    parts = path.split("/")
    renamed = [rename.get(p, p) if i < len(parts) - 1 else p
               for i, p in enumerate(parts)]
    new_path = "/".join(renamed)
    return new_path if new_path != path else path


def _derive_file_path(rel: str, derived_tree: dict) -> str:
    """Map *rel* to its derived-tree path."""
    if not _strip_root(rel, derived_tree.get("file_roots")):
        return rel
    # Renaming applies to all dir components of the path, including the
    # components of the root itself.
    parts = rel.split("/")
    renamed = [derived_tree["rename"].get(p, p) for p in parts[:-1]]
    return "/".join(renamed + [parts[-1]])


def get_ppu_file_path(rel_filepath: str, derived_tree: Optional[dict] = None) -> str:
    """Compute the PPU-equivalent output path for a CUDA source file.

    sailify does NOT rename files or directories by default — the output
    path is the same as the input path.  With a *derived_tree* config a
    renamed directory component mirrors the file into the PPU derived
    tree, so the converted file lands there while the original path keeps
    its pristine copy.
    """
    if not derived_tree:
        return rel_filepath
    rel = rel_filepath.replace(os.sep, "/")
    return _derive_file_path(rel, derived_tree)


def _copy_derived_files(project_directory: str, output_directory: str,
                        derived_tree: Optional[dict]) -> None:
    """Mirror derived_tree["copy"] files verbatim into the derived tree."""
    patterns = (derived_tree or {}).get("copy")
    if not patterns or not output_directory:
        return
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(project_directory, pattern),
                                   recursive=True))
        if not matches:
            raise ValueError(
                f'derived_tree "copy" pattern matched nothing: {pattern!r}'
            )
        for path in matches:
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, project_directory).replace(os.sep, "/")
            derived_rel = _derive_file_path(rel, derived_tree)
            dst = os.path.join(output_directory, derived_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(path, dst)


def _mask_comments_and_strings(source: str, mask_strings: bool = False) -> str:
    """Mask comments and char literals with 'x' to gate identifier replacement.

    String-literal contents stay unmasked by default (the CUDA identifier
    passes have always rewritten them); *mask_strings* additionally masks
    string contents — the sail:: namespace rewriter uses it so prose like
    error messages keeps the stock spellings.
    """
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

        if in_include[i]:
            result.append(c)
            i += 1
            continue

        if in_context == '':
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
                result.append(c)
                i += 1
                if i < n:
                    result.append('x' if mask_strings else source[i])
                    i += 1
                continue
            if c == '"':
                in_context = ''
            result.append('x' if mask_strings else c)
            i += 1

        elif in_context == "'":
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
                    result.append(source[j])
                i = n
            else:
                for j in range(i, close_pos):
                    result.append(source[j])
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


def _rewrite_include_directives(source: str, derived_tree: Optional[dict] = None) -> str:
    """Apply CUDA_INCLUDE_MAP path replacements specifically in #include directives."""
    from .cuda_to_ppu_mappings import CUDA_INCLUDE_MAP

    _inc_re = re.compile(
        r'^([ \t]*#[ \t]*include[ \t]+[<"])([^>"]+)([>"])',
        re.MULTILINE,
    )

    _sorted_pairs = sorted(CUDA_INCLUDE_MAP.items(), key=lambda kv: -len(kv[0]))

    def _replace(match):
        prefix = match.group(1)
        path = match.group(2)
        suffix = match.group(3)

        new_path = path

        # Derived-tree rewrite runs first, so renamed paths are not mangled
        # by the bare CUDA include keys below (e.g. a "cuda/..." key
        # matching inside an already-renamed path).
        if derived_tree:
            new_path = _derive_tree_segments(
                new_path, derived_tree.get("include_roots"), derived_tree["rename"]
            )

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
                if filename == cuda_path:
                    if last_slash >= 0:
                        new_path = new_path[:last_slash + 1] + ppu_path
                    else:
                        new_path = ppu_path

        # SDK-nested cccl families (wt_v3 layout): prefix with sail/ so the
        # include resolves under <SDK>/include/sail/.  Inactive on
        # flat-layout SDKs (_SAIL_INCLUDE_FAMILIES stays empty).
        for fam in _SAIL_INCLUDE_FAMILIES:
            if (new_path == fam or new_path.startswith(fam + "/")
                    or new_path.startswith(fam + ".")):
                new_path = "sail/" + new_path
                break

        # CUDA core headers the SDK ships only as hggc_* (wt_v3 layout).
        # Same filename-component matching as the bare CUDA_INCLUDE_MAP
        # keys above, so subdirectory includes keep their directory.
        for bare, prefixed in _PREFIXED_INCLUDE_MAP.items():
            last_slash = new_path.rfind('/')
            filename = new_path[last_slash + 1:] if last_slash >= 0 else new_path
            if filename == bare:
                new_path = (new_path[:last_slash + 1] if last_slash >= 0
                            else "") + prefixed
                break
        return prefix + new_path + suffix

    return _inc_re.sub(_replace, source)


# ── Preprocessor (per-file) ───────────────────────────────────────────

def _ppu_repl(match):
    """Regex replacement callback."""
    word = match.group(1)
    if word in _CUSTOM_SPECIAL_MAP:
        return _CUSTOM_SPECIAL_MAP[word]
    return _PPU_MAP.get(word, word)


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
    derived_tree: Optional[dict] = None,
) -> SailifyResult:
    """Process a single CUDA source file and write the PPU equivalent.

    With *inject_compat_include*, header files (.h/.hpp/.cuh) additionally get
    the compat wrapper include (*compat_include_line*) prepended.

    *inject_compat_include_ignores* — glob patterns (relative to
    *project_directory*) of headers to exempt from the injection.
    """
    if project_directory:
        rel = os.path.relpath(filepath, project_directory)
    else:
        rel = os.path.relpath(filepath, os.path.dirname(filepath))
    ppu_rel = get_ppu_file_path(rel, derived_tree=derived_tree)
    _is_derived_write = (
        bool(derived_tree) and ppu_rel != rel.replace(os.sep, "/"))
    if output_directory:
        out_path = os.path.join(output_directory, ppu_rel)
    else:
        out_path = os.path.join(os.path.dirname(filepath), ppu_rel)

    result = SailifyResult(CurrentState.INITIALIZED, os.path.abspath(out_path))

    # In-place derived-tree mode: only files that map to a renamed derived
    # path may be written; everything else stays pristine and is reported
    # as "skipped".
    _derive_in_place = (
        bool(derived_tree)
        and _is_derived_write is False
        and project_directory
        and os.path.abspath(output_directory or os.path.dirname(filepath))
        == os.path.abspath(project_directory))
    if _derive_in_place:
        result.status = "skipped"
        return result

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
    output_source = _rewrite_include_directives(output_source, derived_tree=derived_tree)

    # 2. Apply main CUDA→PPU mappings via the pre-compiled trie regex
    _masked = _mask_comments_and_strings(output_source)

    _new_parts = []
    _last_end = 0
    for _m in _RE_PPU_PREPROCESSOR.finditer(output_source):
        if _masked[_m.start()] == 'x':
            continue
        _new_parts.append(output_source[_last_end:_m.start()])
        _new_parts.append(_ppu_repl(_m))
        _last_end = _m.end()
    _new_parts.append(output_source[_last_end:])
    output_source = "".join(_new_parts)

    # 2b. Apply the SDK >= 2.2 identifier group (enabled when the detected
    # SDK version is 2.2+)
    if _SDK22_MAP and _SDK22_RE is not None:
        _masked22 = _mask_comments_and_strings(output_source)
        _new_parts = []
        _last_end = 0
        for _m in _SDK22_RE.finditer(output_source):
            if _masked22[_m.start()] == 'x':
                continue
            _new_parts.append(output_source[_last_end:_m.start()])
            _new_parts.append(_SDK22_MAP.get(_m.group(1), _m.group(1)))
            _last_end = _m.end()
        _new_parts.append(output_source[_last_end:])
        output_source = "".join(_new_parts)

    # 2c. Namespace-resolve the cccl families the SDK nests under sail/
    # (sail::thrust, sail::cub, ...); inactive on flat-layout SDKs.  The
    # combined regex only fires in namespace contexts (fam::, ::fam,
    # namespace fam, namespace x = fam;), so ordinary identifiers named
    # `cub`/`thrust` (actlize loop variables) pass through untouched.
    # Strings are masked here so error-message prose keeps stock spellings.
    # Project-specific post-trie fixes (e.g. torch's at_cuda_detail cub
    # wrapper idioms) belong in the extra-mapping list applied below.
    if _SAIL_NS_COMBINED_RE is not None:
        _masked_ns = _mask_comments_and_strings(output_source, mask_strings=True)
        output_source = _sub_masked(
            _SAIL_NS_COMBINED_RE, _sail_ns_repl, output_source, _masked_ns
        )

    for _pat, _repl in _EXTRA_MAPPING_LISTS:
        output_source = output_source.replace(_pat, _repl)

    if (
        inject_compat_include
        and os.path.splitext(filepath)[1] in DEFAULT_HEADER_EXTENSIONS
        and not any(_matches_pattern(rel.replace(os.sep, "/"), pat)
                    for pat in inject_compat_include_ignores)
    ):
        output_source = _inject_compat_include(output_source, compat_include_line)

    # Derived-only mappings apply only to files written into the derived tree.
    if _is_derived_write:
        for _pat, _repl in _DERIVED_EXTRA_MAPPING_LISTS:
            output_source = output_source.replace(_pat, _repl)

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
    derived_tree: Optional[dict] = None,
    skip: Optional[list] = None,
) -> dict:
    """Convert CUDA code to PPU code in the given project directory.

    *ppu_sdk_version* — explicit (major, minor[, patch]) PPU SDK tuple for
    the version-conditional fixups.  When None it is auto-detected from the
    environment (see ppu_sdk_version.detect_ppu_sdk_version).

    *inject_compat_include* — prepend the compat wrapper include to every
    converted header (.h/.hpp/.cuh).  The path segment follows the name of
    *ppu_compat_dir* (default ``.ppu_compat``); consumers must put the PARENT
    of the compat dir on their include search path.

    *inject_compat_include_ignores* — glob patterns (relative to
    *project_directory*) of headers exempt from the injection.

    *skip* — project-relative paths of files to leave verbatim (still
    mirrored to the output tree by the out-of-place copy).  Meant for
    shared headers whose existing guards already handle the target
    backend.  Reported with status "skipped".

    *derived_tree* — keep the original tree and write converted files with
    renamed directory components into a mirrored derived tree.  Works
    out-of-place (mirrored tree) and in place, where it turns into
    "derive only": just the renamed mirrors are written while every other
    file stays pristine.  The caller configures the mirroring:

        {"rename": {"<src-dir>": "<dst-dir>"},
         "file_roots": [...],     # optional, project-relative roots scoping
                                  # the mirroring
         "include_roots": [...],  # optional, include-path roots
         "copy": [...]}           # optional, glob patterns mirrored verbatim

    Files with a renamed directory component are written to the mirrored
    tree instead, so the output tree contains both the pristine original
    and the converted backend.
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
        _out_of_place = False
        if output_directory:
            output_directory = os.path.abspath(output_directory)
            _out_of_place = os.path.abspath(output_directory) != project_directory
            if _out_of_place:
                _copytree_compat(project_directory, output_directory)
        else:
            output_directory = project_directory
        # Out-of-place: the pristine original sits beside the converted
        # tree.  In place: preprocessor() writes only the renamed mirrors
        # and reports everything else as "skipped".
        derived_tree = _normalize_derived_tree(derived_tree)

        compat_dir = ppu_compat_dir or os.path.join(output_directory, ".ppu_compat")
        include_line = compat_wrapper_include_line(compat_dir)
        _compat_dir_name = os.path.basename(os.path.normpath(compat_dir))
        if _compat_dir_name not in SKIP_DIRS:
            SKIP_DIRS.add(_compat_dir_name)
            _added_skip_dirs.append(_compat_dir_name)

        _includes = includes or ["*"]
        _ignores = ignores or []
        _skip_rel = {
            s.replace("\\", "/").strip("/") for s in skip
        } if skip else set()

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
        _maybe_enable_sdk_2_2_mappings(ppu_sdk_version)
        if install_ppu_compat:
            runtime_api_version = detect_runtime_api_version()
            _print_detected_versions(ppu_sdk_version, runtime_api_version)
            write_compat_wrapper_header(
                compat_dir,
                version_config if version_config else None,
                ppu_sdk_version=ppu_sdk_version,
                runtime_api_version=runtime_api_version,
            )
        elif inject_compat_include:
            log.warning(
                "inject_compat_include is on but install_ppu_compat is off: converted "
                "headers will reference %s which is not generated in this run.",
                include_line,
            )

        results = {}

        for filepath in all_files:
            if _skip_rel and os.path.relpath(
                filepath, project_directory
            ).replace(os.sep, "/") in _skip_rel:
                res = SailifyResult(ppuified_path=os.path.abspath(filepath))
                res.status = "skipped"
                results[os.path.abspath(filepath)] = res
                continue
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
                derived_tree=derived_tree,
            )
            results[os.path.abspath(filepath)] = res

        _copy_derived_files(project_directory, output_directory, derived_tree)

        if verbose:
            n_ok = sum(1 for r in results.values() if r.status == "ok")
            print(f"sailify: done. {n_ok}/{len(results)} source files converted.")

    finally:
        for d in _added_skip_dirs:
            SKIP_DIRS.discard(d)

    return results


def _ensure_ppu_compat(output_directory: str) -> None:
    """(Re)generate .ppu_compat in output_directory with the detected PPU SDK version."""
    compat_dir = os.path.join(output_directory, ".ppu_compat")
    ppu_sdk_version = detect_ppu_sdk_version()
    _maybe_enable_sdk_2_2_mappings(ppu_sdk_version)
    runtime_api_version = detect_runtime_api_version()
    _print_detected_versions(ppu_sdk_version, runtime_api_version)
    write_compat_wrapper_header(compat_dir, ppu_sdk_version=ppu_sdk_version,
                                runtime_api_version=runtime_api_version)


def sailify_extra_files(
    output_directory: str,
    extra_files: list,
    show_detailed: bool = False,
    install_ppu_compat: bool = True,
    inject_compat_include: bool = False,
    inject_compat_include_ignores: list = None,
) -> dict:
    """Sailify only the specified extra_files.

    Converts a list of user-provided CUDA source files in-place and returns
    a dict mapping absolute input path -> SailifyResult.
    """
    output_directory = os.path.abspath(output_directory)
    _maybe_enable_sdk_2_2_mappings(detect_ppu_sdk_version())
    if install_ppu_compat:
        _ensure_ppu_compat(output_directory)
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
    _maybe_enable_sdk_2_2_mappings(detect_ppu_sdk_version())
    if install_ppu_compat:
        _ensure_ppu_compat(output_directory)
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

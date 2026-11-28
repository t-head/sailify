"""Ground-truth check of the SDK-versioned mappings against the installed
PPU SDK headers.  Destination symbols that exist nowhere in the SDK include
tree are exactly the class of bug code review kept finding (plausible-looking
names that no header defines).  Skips when no PPU SDK is available.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from sailify.cuda_to_ppu_mappings import (  # noqa: E402
    PPU_SDK_2_3_IDENTIFIER_MAP,
    PPU_SDK_2_3_RENAME_MAP,
    PPU_SDK_2_3_INCLUDE_PREFIX_MAP,
)

IDENT = re.compile(r"^\w+$")
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


def _sdk_include_roots(ppu_sdk):
    roots = []
    for sub in ("targets/x86_64-linux/include", "include"):
        p = os.path.join(ppu_sdk, sub)
        if os.path.isdir(p):
            roots.append(p)
    return roots


def _collect_header_tokens(roots):
    tokens = set()
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith((".h", ".hpp", ".cuh")):
                    try:
                        with open(os.path.join(dirpath, f), errors="replace") as fh:
                            tokens.update(_TOKEN_RE.findall(fh.read()))
                    except OSError:
                        pass
    return tokens


def main():
    ppu_sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_PATH") or ""
    roots = _sdk_include_roots(ppu_sdk) if ppu_sdk else []
    if not roots:
        print("MAPPING_INTEGRITY_SUMMARY pass=0 fail=0 skip=1 "
              "(no PPU SDK; set PPU_SDK)")
        return 0
    tokens = _collect_header_tokens(roots)

    def _dest_present(dest):
        if dest in tokens:
            return True
        # X-macro families (SANITIZER_HGGC_CBID_*_API_##name) define their
        # identifiers by token pasting: validate the pieces instead.
        if dest.startswith("SANITIZER_HGGC_CBID_") and "_API_" in dest:
            head, _, suffix = dest.partition("_API_")
            return (head + "_API_") in tokens and suffix in tokens
        return False

    npass = 0
    failures = []
    checked = 0
    for dest in PPU_SDK_2_3_IDENTIFIER_MAP.values():
        if not IDENT.match(dest):
            continue  # namespaces (sail::thrust) and friends
        checked += 1
        if _dest_present(dest):
            npass += 1
        else:
            failures.append(dest)

    # Include destinations must resolve to real files under the SDK tree.
    for dest in PPU_SDK_2_3_RENAME_MAP.values():
        checked += 1
        found = any(os.path.isfile(os.path.join(r, dest)) for r in roots)
        if found:
            npass += 1
        else:
            failures.append(dest + " (missing file)")

    # Prefix destinations must resolve to real directories.
    for dest in PPU_SDK_2_3_INCLUDE_PREFIX_MAP.values():
        checked += 1
        found = any(os.path.isdir(os.path.join(r, dest.rstrip("/"))) for r in roots)
        if found:
            npass += 1
        else:
            failures.append(dest + " (missing directory)")

    for d in failures:
        print("FAIL integrity destination not in SDK headers: " + d)
    print("MAPPING_INTEGRITY_SUMMARY pass=%d fail=%d skip=0 (checked=%d)"
          % (npass, len(failures), checked))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

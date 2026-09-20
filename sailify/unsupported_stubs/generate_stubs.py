"""Auto-generate C/C++ stub implementations for unsupported CUDA APIs.

This module reads ``unsupported_apis.json`` and generates a C
header file (``ppu_unsupported_stubs.h``) containing ``static inline`` stub
functions.  Each stub prints an error message and calls ``exit(1)``.

The JSON config already contains PPU-mapped function names and types —
this module simply formats them into C code.  An API may carry an optional
``version`` range (``{"begin": .., "end": ..}`` in ``PPU_SDK_VERSION`` units,
major*10000+minor*100+patch) restricting the stub to those SDK versions;
without it the stub is defined for every version.

Usage from sailify_python.py::

    from .unsupported_stubs import write_stubs_header
    write_stubs_header(compat_dir)
"""

import json
import os
import logging

log = logging.getLogger(__name__)


def _format_param(decl_template: str, name: str) -> str:
    """Format a parameter declaration from a type_str template.

    The type_str may contain ``{}`` as a placeholder for the variable name
    (e.g. ``"const void * {}"``).  If no placeholder is present, the name
    is appended.
    """
    if '{}' in decl_template:
        return decl_template.format(name)
    return f"{decl_template} {name}"


def _version_gate(api: dict) -> str:
    """``#if`` line limiting a stub to its unsupported SDK-version range.

    PPU_SDK_VERSION == 0 (unknown) is included so the legacy behavior of
    defining every stub is kept when the SDK version could not be detected.
    """
    ver = api.get("version") or {}
    begin = ver.get("begin")
    end = ver.get("end")
    if begin is None and end is None:
        return ""
    conds = ["PPU_SDK_VERSION == 0"]
    if begin is not None and end is not None:
        conds.append(f"(PPU_SDK_VERSION >= {int(begin)} && PPU_SDK_VERSION <= {int(end)})")
    elif begin is not None:
        conds.append(f"PPU_SDK_VERSION >= {int(begin)}")
    else:
        conds.append(f"PPU_SDK_VERSION <= {int(end)}")
    return "#if " + " || ".join(conds)


def _generate_stub_function(api: dict) -> str:
    """Generate a single ``static inline`` stub function for *api*.

    *api* is a dict with PPU-mapped keys: name, return_type, parameters, library.
    """
    func_name = api["name"]
    return_type = api["return_type"].strip()
    params = api["parameters"]

    # Build parameter list
    param_decls = []
    param_names = []
    for p in params:
        decl = _format_param(p["type"], p["name"])
        param_decls.append(decl)
        param_names.append(p["name"])

    if param_decls:
        params_str = ", ".join(param_decls)
    else:
        params_str = "void"
    decl = f"{return_type} {func_name}({params_str})"

    # Build the stub body
    body_lines = []

    if param_names:
        void_casts = "; ".join(f"(void){n}" for n in param_names)
        body_lines.append(f"    {void_casts};")

    body_lines.append(f'    fprintf(stderr, "{func_name} is not supported.\\n");')
    body_lines.append(f"    exit(1);")

    if return_type and return_type != "void":
        body_lines.append(f"    return ({return_type})0;  /* unreachable */")

    return f"static inline {decl} {{\n" + "\n".join(body_lines) + "\n}"


def generate_stubs_header() -> str:
    """Generate the full content of ``ppu_unsupported_stubs.h``.

    Returns:
        The header file content as a string, ready to be written to disk.
    """
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unsupported_apis.json",
    )
    with open(json_path) as f:
        data = json.load(f)

    apis = data["apis"]
    unmapped_types = data.get("unmapped_types", [])

    # Group by library for readability
    apis_by_lib: dict = {}
    for api in apis:
        lib = api["library"]
        apis_by_lib.setdefault(lib, []).append(api)

    lines = []
    lines.append("#ifndef PPU_UNSUPPORTED_STUBS_H")
    lines.append("#define PPU_UNSUPPORTED_STUBS_H")
    lines.append("")

    for lib_name in sorted(apis_by_lib.keys()):
        lib_apis = apis_by_lib[lib_name]
        lines.append(f"/* ── {lib_name}: {len(lib_apis)} unsupported APIs ── */")
        lines.append("")

        for api in lib_apis:
            gate = _version_gate(api)
            if gate:
                lines.append(gate)
            lines.append(_generate_stub_function(api))
            if gate:
                lines.append("#endif")
            lines.append("")

    lines.append("#endif /* PPU_UNSUPPORTED_STUBS_H */")

    return "\n".join(lines)


def write_stubs_header(output_dir: str) -> str:
    """Write ``ppu_unsupported_stubs.h`` to *output_dir*.

    Args:
        output_dir: Directory to write the header file to (e.g. ``.ppu_compat/``).

    Returns:
        The absolute path to the written header file.
    """
    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "ppu_unsupported_stubs.h")
    content = generate_stubs_header()
    with open(header_path, "w") as f:
        f.write(content)
    stub_count = content.count("static inline")
    log.info("Generated ppu_unsupported_stubs.h (%d stubs) in %s", stub_count, output_dir)
    return header_path

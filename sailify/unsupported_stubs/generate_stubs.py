"""Generate ``ppu_unsupported_stubs.h``: ``static inline`` stubs that print an
error and ``exit(1)`` for APIs the PPU SDK does not implement.

``unsupported_apis.json`` is the superset of possibly-unsupported APIs.
With ``stub_probe=True`` (default) the installed SDK's exported symbols are
probed (see ``probe.py``) and a stub is emitted only for APIs no host library
provides.  Libraries that cannot be probed keep their stubs with their
``version`` begin/end gate, and so does everything when ``stub_probe=False``.

When ``SAILIFY_STUB_PROBE_REPORT`` is set, the APIs found already supported
(stubs dropped) are printed to stdout.
"""

import json
import os
import logging

from .probe import probe_supported_apis

log = logging.getLogger(__name__)

_REPORT_ENV = "SAILIFY_STUB_PROBE_REPORT"


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


def _print_supported_report(apis, probe):
    if not os.environ.get(_REPORT_ENV):
        return
    supported = sorted(a["name"] for a in apis if a["name"] in probe.supported)
    print(f"sailify stub probe: {len(supported)}/{len(apis)} APIs already "
          f"supported by this SDK (stubs dropped):")
    for name in supported:
        print(f"  {name}")


def generate_stubs_header(stub_probe=True):
    """Generate the full content of ``ppu_unsupported_stubs.h``.

    ``stub_probe`` True (default) probes the installed SDK and stubs only
    APIs it does not export; libraries that cannot be probed keep their
    version-gated stubs.  False emits every JSON entry with its version gate.
    """
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unsupported_apis.json",
    )
    with open(json_path) as f:
        data = json.load(f)

    apis = data["apis"]

    probe = probe_supported_apis(apis) if stub_probe else None
    if probe is not None:
        _print_supported_report(apis, probe)

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

        probed = probe is not None and lib_name in probe.probed
        for api in lib_apis:
            if probed and api["name"] in probe.supported:
                continue
            # The probe is specific to the concrete SDK; unprobed libraries
            # fall back to the compile-time version gate.
            gate = "" if probed else _version_gate(api)
            if gate:
                lines.append(gate)
            lines.append(_generate_stub_function(api))
            if gate:
                lines.append("#endif")
            lines.append("")

    lines.append("#endif /* PPU_UNSUPPORTED_STUBS_H */")

    return "\n".join(lines)


def write_stubs_header(output_dir: str, stub_probe: bool = True) -> str:
    """Write ``ppu_unsupported_stubs.h`` to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "ppu_unsupported_stubs.h")
    content = generate_stubs_header(stub_probe=stub_probe)
    with open(header_path, "w") as f:
        f.write(content)
    log.info("Generated ppu_unsupported_stubs.h (%d stubs) in %s",
             content.count("static inline"), output_dir)
    return header_path

# AGENTS.md

sailify converts CUDA source code to HGGC code and generates `.ppu_compat/` headers. It does not rename project files/directories or rewrite CMake/Python build scripts.

## Core rules

- Main-repository mappings must have sources from CUDA source code; destinations must refer to PPU_SDK equivalents or sailify-provided stubs/compatibility identifiers (such as `compatible*`). Project-specific renames and other non-SDK mappings belong in the user's own repository, supplied through `--extra-mapping` JSON.
- Preserve conversion idempotency: converting already-converted source again must not change it.
- Maintain compatibility with **all released PPU_SDK versions starting from 2.1**. New mappings, stubs, shim retirement, and default changes must be scoped to the relevant SDK versions without regressing older releases.
- Compatibility headers installed into `.ppu_compat/` provide only stubs, include forwarding, and compatibility glue, except for the compile-time fixup layer in `sailify/ppu_sdk_fixups.h`. Implementing missing APIs' real functionality belongs in PPU_SDK, not sailify.
- Follow the existing Python style: four-space indentation; English code, comments, and documentation.

## Where changes belong

Extend existing data/configuration for cases the current machinery already handles; change Python logic for new behavior. Python mapping tables are also data: do not add per-API branches to the engine or generator when an existing table can express the change.

| Change | Location |
|---|---|
| identifier/type/header/API mappings | `sailify/cuda_to_ppu_mappings.py` |
| nvcc option mappings and translation behavior | `sailify/nvcc_hgcc_options.py` |
| Stub signatures, version ranges | `sailify/unsupported_stubs/unsupported_apis.json` (`apis`) |
| Custom stub bodies for selected APIs | `sailify/unsupported_stubs/custom_impl.json` |
| Stub generation or SDK-library probing behavior | `sailify/unsupported_stubs/generate_stubs.py`, `probe.py` |
| Conversion, mapping activation, compatibility-header generation/installation | `sailify/sailify_python.py` |
| SDK/runtime version detection | `sailify/ppu_sdk_version.py` |
| CLI/config option handling | `sailify_cli.py` |

Keep project-specific settings in the user's config/extra mappings rather than hardcoding them in sailify. New public options should stay consistent across CLI, config JSON, and applicable Python APIs.

## SDK release adaptation

- **PPU_SDK update:** compare headers/APIs with older releases; review mappings, version detection, fixups, and `.ppu_compat/` installation. When the SDK supplies a formerly missing header, stop installing/using its shim for that SDK, but retain it for older releases that need it. Do not remove still-needed include mappings just because a shim retires; check re-generated output for stale shims that could shadow SDK headers.
- **Stub database:** retain the full maintained stub set when PPU_SDK gains APIs; do not delete entries merely because the newest SDK supports them. Probing selects stubs for the installed SDK; keep version-gated fallback behavior correct when probing is disabled or unavailable.
- **new identifier/type/header/API/option update:** check new identifiers, types, headers, APIs, and nvcc options. Extend mappings and add missing stubs as needed, with valid signatures/types and not-supported results rather than functional implementations. Review compatible-version defaults without silently changing older SDK behavior.
- Use the relevant SDK version for each decision. Compile-time PPU fixups use `PPU_SDK_VERSION` (`major * 10000 + minor * 100 + patch`).
- When shipped compatibility assets change, check both their packaging in `setup.py` and their installation into `.ppu_compat/`; editing a header alone may not be sufficient.
- After completing adaptation to a new SDK release, bump the sailify version number.

## Development commands

Run from the repository root:

```bash
python -m pip install -e .
python tests/api_patterns_test.py
python tests/probe_test.py
PPU_SDK=/path/to/PPU_SDK bash tests/run_tests.sh
```

The API-pattern smoke test needs no PPU SDK. Probe tests use a simulated SDK. The full suite requires an installed SDK with `envsetup.sh` and `hgcc`; `--skip-compile` does not remove that prerequisite.

The full suite recreates `${TMPDIR:-/tmp}/sailify_tests_run`; use a separate `TMPDIR` for concurrent runs. Logs are under that directory's `logs/`.

## Verification

- Add focused regressions for changed mappings, conversion, probing, or generation; add CUDA-source compile cases under `tests/cases/compat/<category>/` when compiler validation is needed, with `flags.txt` for extra flags.
- Gate SDK-dependent cases by the versions they require. Extend the test runner's gating logic as needed; a new version-list file alone does not activate a gate.
- For SDK adaptation, validate the new release and older affected releases, including shim retention/retirement and probing enabled/disabled/unavailable. Passing on the newest SDK alone is insufficient.
- Run smoke tests and relevant targeted tests; run the full suite for conversion/header changes when an SDK is available. Report tested SDK versions and any untested paths explicitly; do not claim full release coverage from a partial matrix.

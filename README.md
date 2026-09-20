# SAILIFY

SAILIFY is a set of tools that you can use to automatically translate CUDA source code into HGGC code.

## Installation

```bash
cd /path/to/sailify
pip install -e .
```

## How to Use

### CLI

```bash
# Basic conversion (in-place)
sailify /path/to/cuda/project

# Out-of-place conversion
sailify /path/to/cuda/project --output-directory /path/to/output

# With excludes and custom versions
sailify /path/to/cuda/project \
  --output-directory /path/to/output \
  --excludes third_party tests \
  --cuda-version 13.0.0 \
  --cudnn-version 8.9.5 \
  --nccl-version 2.27.3

# With extra string-to-string mappings
sailify /path/to/cuda/project --extra-mapping extra_patterns.json

# Using a config file
sailify --config-json config.json
```

#### CLI Options

| Option | Description |
|---|---|
| `project_directory` | Root directory of the CUDA project to convert |
| `--output-directory` | Where to write converted files (default: in-place) |
| `--excludes` | Directories to exclude (relative to project root) |
| `--includes` | Glob patterns for files to include |
| `--ignores` | Glob patterns for files to ignore |
| `--custom-map-json` | Path to JSON file with extra CUDA→PPU identifier mappings |
| `--extra-mapping` | Path to JSON file with extra string-to-string mappings (array of `[pattern, replacement]` pairs) |
| `--config-json` | Path to a JSON config file with all options |
| `--verbose` / `-v` | Enable verbose logging |
| `--backup` | Create .bak backup files before in-place conversion |
| `--no-ppu-compat` | Skip generating .ppu_compat/ compat headers |
| `--ppu-compat-dir` | Custom directory for .ppu_compat headers (default: `<output>/.ppu_compat`) |
| `--cuda-version` | CUDA version for COMPATIBLE_VERSION macro |
| `--cublas-version` | cuBLAS version |
| `--cufft-version` | cuFFT version |
| `--curand-version` | cuRAND version |
| `--cusparse-version` | cuSPARSE version |
| `--cusolver-version` | cuSOLVER version |
| `--cudnn-version` | cuDNN version |
| `--nccl-version` | NCCL version |
| `--video-version` | Video Codec SDK version |
| `--cupti-version` | CUPTI API version number |
| `--npp-version` | NPP version |
| `--ppu-sdk-version` | PPU SDK version for conditional fixups (e.g. `2.2` or `2v2`). Auto-detected when omitted |

When no `--*-version` option is given, the `COMPATIBLE_*` defaults follow the
detected hggcrt Runtime API version: `Runtime API version vN` from
`hgcc --version`, else an `hggcrt_version:vN` line in `$PPU_PATH/VERSION.txt`,
else v3.

### Python API

```python
from sailify.sailify_python import sailify

results = sailify(
    project_directory="/path/to/cuda/project",
    output_directory="/path/to/output",
    excludes=["third_party"],
    version_config={"cuda": "13.0.0", "cudnn": "8.9.5"},
    install_ppu_compat=True,
    verbose=True,
)
```

Convert specific files:

```python
from sailify.sailify_python import sailify_extra_files

results = sailify_extra_files(
    output_directory="/path/to/output",
    extra_files=["/path/to/extension.cu", "/path/to/kernel.cuh"],
)
```

Convert with recursive dependency discovery:

```python
from sailify.sailify_python import sailify_extra_files_recursive

results = sailify_extra_files_recursive(
    output_directory="/path/to/output",
    extra_files=["/path/to/extension.cu"],
    header_include_dirs=["/path/to/includes"],
)
```

## .ppu_compat Directory

When `install_ppu_compat=True` (default), sailify generates compatibility headers (including `compatible_wrapper.h`, PPU SDK fixups, unsupported API stubs, etc.) in the `.ppu_compat/` directory. When compiling the converted project, add this path to the include search path and force-include the main wrapper header:

```bash
hgcc -I/path/to/output/.ppu_compat -include compatible_wrapper.h ...
```

### PPU SDK Version Detection

`ppu_sdk_fixups.h` contains shims whose behavior depends on the PPU SDK release. sailify resolves the SDK version when generating `.ppu_compat/` and bakes it into `compatible_wrapper.h` as `PPU_SDK_VERSION` (`major * 10000 + minor * 100 + patch`, e.g. 2.1 → 20100, 2.1.1 → 20101, 2.2 → 20200; 0 = unknown):

1. `--ppu-sdk-version` (CLI) / `ppu_sdk_version` (config JSON / `sailify()` keyword) — explicit, wins over everything.
2. `hgcc --version` — the compiler is located via `hgcc` on `PATH`, then `$PPU_SDK/bin`, then `$PPU_HOME/bin`.

If none succeed, `PPU_SDK_VERSION` is 0 and the legacy fixup behavior is kept; a warning is logged at conversion time. The macro can also be overridden at compile time with `-DPPU_SDK_VERSION=...`.

## License

Apache License 2.0

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

## License

Apache License 2.0

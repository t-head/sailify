#!/usr/bin/env python3
"""
sailify_cli — Command-line interface for generic CUDA→PPU source conversion.

Usage:
    sailify /path/to/cuda/project
    sailify /path/to/cuda/project --output-directory /path/to/output
"""

import os
import sys
import json
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Convert CUDA source code to PPU source code."
    )
    parser.add_argument(
        "project_directory", nargs="?", type=str, default=None,
        help="Root directory of the CUDA project to convert.",
    )
    parser.add_argument(
        "--output-directory", type=str, default=None,
        help="Where to write converted files (default: in-place).",
    )
    parser.add_argument(
        "--excludes", nargs="*", default=[],
        help="Directories to exclude from conversion (relative to project root).",
    )
    parser.add_argument(
        "--includes", nargs="*", default=None,
        help="Glob patterns for files to include.",
    )
    parser.add_argument(
        "--ignores", nargs="*", default=None,
        help="Glob patterns for files to ignore.",
    )
    parser.add_argument(
        "--custom-map-json", type=str, default="",
        help="Path to a JSON file with extra CUDA→PPU mappings.",
    )
    parser.add_argument(
        "--extra-mapping", type=str, default="",
        help="Path to a JSON file with extra string-to-string mappings "
             "(array of [pattern, replacement] pairs).",
    )
    parser.add_argument(
        "--config-json", type=str, default=None,
        help="Path to a JSON config file with all options.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--backup", action="store_true", default=False,
        help="Create .bak backup files before in-place conversion.",
    )
    parser.add_argument(
        "--no-ppu-compat", action="store_false", dest="install_ppu_compat",
        help="Skip generating .ppu_compat/ compat headers.",
    )
    parser.add_argument(
        "--ppu-compat-dir", type=str, default="",
        help="Directory for .ppu_compat headers (default: <output>/.ppu_compat).",
    )
    # Version parameters
    parser.add_argument(
        "--cuda-version", type=str, default=None,
        help="CUDA version for COMPATIBLE_VERSION macro (e.g. 13.0.0).",
    )
    parser.add_argument(
        "--cublas-version", type=str, default=None,
        help="cuBLAS version for COMPATIBLE_BLAS_VERSION macro (e.g. 13.0.0).",
    )
    parser.add_argument(
        "--cufft-version", type=str, default=None,
        help="cuFFT version for COMPATIBLE_FFT_VERSION macro (e.g. 12.0.0).",
    )
    parser.add_argument(
        "--curand-version", type=str, default=None,
        help="cuRAND version for COMPATIBLE_RAND_VERSION macro (e.g. 10.4.0).",
    )
    parser.add_argument(
        "--cusparse-version", type=str, default=None,
        help="cuSPARSE version for COMPATIBLE_SPARSE_VERSION macro (e.g. 12.0.0).",
    )
    parser.add_argument(
        "--cusolver-version", type=str, default=None,
        help="cuSOLVER version for COMPATIBLE_SOLVER_VERSION macro (e.g. 12.0.3).",
    )
    parser.add_argument(
        "--cudnn-version", type=str, default=None,
        help="cuDNN version for COMPATIBLE_DNN_VERSION macro (e.g. 8.9.5).",
    )
    parser.add_argument(
        "--nccl-version", type=str, default=None,
        help="NCCL version for COMPATIBLE_NCCL_VERSION macro (e.g. 2.27.3).",
    )
    parser.add_argument(
        "--video-version", type=str, default=None,
        help="Video Codec SDK version for COMPATIBLE_VIDEO_VERSION macro (e.g. 13.0.19).",
    )
    parser.add_argument(
        "--cupti-version", type=int, default=None,
        help="CUPTI API version number for COMPATIBLE_CUPTI_VERSION macro (e.g. 130000).",
    )
    parser.add_argument(
        "--npp-version", type=str, default=None,
        help="NPP version for compatiblePpGetLibVersion (e.g. 13.0.50).",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load from config JSON if provided
    project_directory = args.project_directory
    output_directory = args.output_directory
    excludes = args.excludes
    includes = args.includes
    ignores = args.ignores
    custom_map_json = args.custom_map_json
    extra_mapping = args.extra_mapping
    backup = args.backup
    version_config = {}

    if args.config_json:
        if not os.path.isfile(args.config_json):
            print(f"ERROR: Config file not found: {args.config_json}", file=sys.stderr)
            sys.exit(1)
        with open(args.config_json) as fh:
            cfg = json.load(fh)

        config_dir = os.path.dirname(os.path.abspath(args.config_json))
        if project_directory is None:
            project_directory = cfg.get("project_directory")
        if project_directory:
            project_directory = os.path.join(config_dir, project_directory)
        if "output_directory" in cfg and cfg["output_directory"]:
            output_directory = os.path.join(config_dir, cfg["output_directory"])
        excludes = cfg.get("excludes", [])
        includes = cfg.get("includes")
        ignores = cfg.get("ignores")
        backup = cfg.get("backup", backup)
        if "custom_map_json" in cfg and cfg["custom_map_json"]:
            custom_map_json = os.path.join(config_dir, cfg["custom_map_json"])
        if "extra_mapping" in cfg and cfg["extra_mapping"]:
            extra_mapping = os.path.join(config_dir, cfg["extra_mapping"])
        if "version_config" in cfg and cfg["version_config"]:
            version_config.update(cfg["version_config"])

    # Build version config from CLI args (overrides JSON)
    if args.cuda_version:
        version_config["cuda"] = args.cuda_version
    if args.cublas_version:
        version_config["cublas"] = args.cublas_version
    if args.cufft_version:
        version_config["cufft"] = args.cufft_version
    if args.curand_version:
        version_config["curand"] = args.curand_version
    if args.cusparse_version:
        version_config["cusparse"] = args.cusparse_version
    if args.cusolver_version:
        version_config["cusolver"] = args.cusolver_version
    if args.cudnn_version:
        version_config["cudnn"] = args.cudnn_version
    if args.nccl_version:
        version_config["nccl"] = args.nccl_version
    if args.video_version:
        version_config["video"] = args.video_version
    if args.cupti_version is not None:
        version_config["cupti"] = args.cupti_version
    if args.npp_version:
        version_config["npp"] = args.npp_version
    if not version_config:
        version_config = None

    if not project_directory:
        parser.error("project_directory or --config-json is required.")

    project_directory = os.path.abspath(project_directory)
    if not os.path.isdir(project_directory):
        print(f"ERROR: Project directory not found: {project_directory}", file=sys.stderr)
        sys.exit(1)

    # Run conversion
    from sailify.sailify_python import sailify

    results = sailify(
        project_directory=project_directory,
        output_directory=output_directory or "",
        excludes=excludes,
        includes=includes,
        ignores=ignores,
        custom_map_list=custom_map_json,
        extra_mapping=extra_mapping,
        version_config=version_config,
        backup=backup,
        install_ppu_compat=args.install_ppu_compat,
        ppu_compat_dir=args.ppu_compat_dir or "",
        verbose=args.verbose,
    )

    # Summary
    n_ok = sum(1 for r in results.values() if r.status == "ok")
    n_skip = sum(1 for r in results.values() if r.status == "skipped")
    print(f"sailify: {n_ok} files converted, {n_skip} files skipped.")


if __name__ == "__main__":
    main()

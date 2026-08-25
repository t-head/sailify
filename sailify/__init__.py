"""
sailify — Generic CUDA to PPU source code conversion tool.

A Python tool for converting CUDA code to PPU code by replacing
CUDA SDK identifiers, types, and API calls with their PPU SDK equivalents.
"""

from .version import __version__
from .nvcc_hgcc_options import NVCC_HGCC_OPTIONS

__all__ = ['__version__', 'NVCC_HGCC_OPTIONS']

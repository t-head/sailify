"""Unsupported API stub generator package.

This package auto-generates C/C++ stub implementations for APIs that
PPU SDK does not implement. The generator (:func:`generate_stubs_header`)
produces a header file (``ppu_unsupported_stubs.h``) that is included
by ``compatible_wrapper.h``.

Public API:
    generate_stubs_header()  → str  : Generate ppu_unsupported_stubs.h content
    write_stubs_header(dir)  → None : Write ppu_unsupported_stubs.h to a directory
"""

from .generate_stubs import generate_stubs_header, write_stubs_header

__all__ = ["generate_stubs_header", "write_stubs_header"]

from __future__ import annotations

"""Compatibility wrapper for historical import path `tfn.core.field_interference_block`.

The implementation of `ImageFieldInterference` lives in
`tfn.core.image.field_interference_block`.  To avoid widespread refactors
and keep old tests working, we re-export it here.
"""

from .image.field_interference_block import ImageFieldInterference  # noqa: F401

__all__ = [
    "ImageFieldInterference",
] 
from __future__ import annotations

"""Compatibility wrapper for historical import path `tfn.core.field_emitter`.

The implementation of `ImageFieldEmitter` lives in
`tfn.core.image.field_emitter`.  This thin module simply re-exports the
class to keep older code functioning.
"""

from .image.field_emitter import ImageFieldEmitter  # noqa: F401

__all__ = [
    "ImageFieldEmitter",
] 
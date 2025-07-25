from __future__ import annotations

"""Compatibility wrapper for historical import path `tfn.core.field_propagator`.

The 2-D image implementation lives in `tfn.core.image.field_propagator`.
We re-export it here so legacy tests continue to import the old name.
"""

from .image.field_propagator import ImageFieldPropagator  # noqa: F401

__all__ = [
    "ImageFieldPropagator",
] 
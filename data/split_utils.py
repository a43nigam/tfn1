from __future__ import annotations

"""Utility helpers for dataset train/val/test splitting.

All loaders should rely on `get_split_sizes` (and keep their own
chronological or random slicing logic) to ensure a single source of
truth for default fractions and validation.
"""
from typing import Dict, Tuple

DEFAULT_SPLIT_FRAC: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def validate_split_frac(split_frac: Dict[str, float]) -> None:
    if not all(k in split_frac for k in ("train", "val", "test")):
        raise ValueError("split_frac must contain 'train', 'val', and 'test' keys")
    total = sum(split_frac.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_frac values must sum to 1.0, got {total}")


def get_split_sizes(n: int, split_frac: Dict[str, float] | None = None) -> Tuple[int, int, int]:
    """Return counts for train, val, test given dataset length ``n``.

    Parameters
    ----------
    n : int
        Total number of samples.
    split_frac : dict | None
        Fractions mapping with keys 'train', 'val', 'test'. Missing or
        ``None`` falls back to :data:`DEFAULT_SPLIT_FRAC`.

    Returns
    -------
    (n_train, n_val, n_test) : tuple of ints summing to *n*.
    """
    frac = split_frac or DEFAULT_SPLIT_FRAC
    validate_split_frac(frac)
    n_train = int(frac["train"] * n)
    n_val = int(frac["val"] * n)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test 
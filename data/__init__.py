import os
from glob import glob

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def resolve_data_path(path: str) -> str:
    """Resolve a data file path that may be located inside Kaggle's input
    directory structure.

    When running inside a Kaggle notebook/competition environment, datasets
    added via the *Add data* sidebar are mounted under ``/kaggle/input`` with
    the directory name corresponding to the dataset slug.  Since those
    directories are read-only and their names may change across user sessions,
    we cannot hard-code absolute paths in config files.

    This helper works as follows:

    1. If *path* already exists (either absolute or relative to the current
       working directory) we simply return it.
    2. Otherwise, we look for a file with the **same basename** (e.g.
       ``ETTh1.csv``) anywhere under ``/kaggle/input`` using ``glob``.  The
       first match is returned.

    The function raises ``FileNotFoundError`` if the file cannot be located so
    that the calling code can fail fast with a clear message.
    """

    # 1. Direct hit – local path already valid
    if os.path.exists(path):
        return path

    # 2. Running inside Kaggle? If so, attempt recursive search under /kaggle/input
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        basename = os.path.basename(path)
        # ** recursive glob needs "**" pattern and recursive=True flag (Python ≥3.5)
        candidates = glob(os.path.join(kaggle_root, "**", basename), recursive=True)
        if candidates:
            return candidates[0]

    # 3. Out of luck
    raise FileNotFoundError(f"resolve_data_path: could not locate '{path}'. "
                            "If you are running on Kaggle, make sure the dataset "
                            "has been added to the *Data* tab and that the file "
                            "name matches the expected basename.")


# Re-export for convenience so callers can do `from data import resolve_data_path`
__all__ = ["resolve_data_path"] 
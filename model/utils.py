"""
model.utils
------------
Centralised helpers shared across training utilities (train.py, hyperparameter_search.py, etc.).

Right now it provides a single public function:
    build_model – instantiate a model from the registry while filtering
                   unsupported parameters.  Keeping this in one place
                   prevents the two main entry-points from diverging.
"""

from __future__ import annotations

from typing import Dict, Any
import torch.nn as nn

from model import registry

__all__ = ["build_model"]


def build_model(model_name: str, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any] | None = None) -> nn.Module:
    """Instantiate *model_name* with *model_cfg*.

    The logic is identical for both *train.py* and *hyperparameter_search.py*;
    we keep it here to avoid code duplication.
    """
    model_info = registry.get_model_config(model_name)
    model_cls = model_info["class"]
    task_type = model_info["task_type"]

    # ---- merge defaults ---------------------------------------------------
    model_args: Dict[str, Any] = dict(model_info.get("defaults", {}))
    model_args.update(model_cfg)

    # ---- copy data-dependent settings -------------------------------------
    if data_cfg is not None:
        if (
            "output_len" in data_cfg
            and "output_len" in model_info.get("required_params", [])
        ):
            model_args["output_len"] = data_cfg["output_len"]

    # Some constructors accept an explicit *task* argument – provide it if so.
    if "task" in model_cls.__init__.__code__.co_varnames:
        model_args["task"] = task_type

    # ---- filter unsupported keys -----------------------------------------
    allowed: set[str] = set(
        model_info.get("required_params", []) + model_info.get("optional_params", [])
    )
    allowed.add("task")  # always allowed when present

    filtered_args: Dict[str, Any] = {}
    dropped: list[str] = []

    for k, v in model_args.items():
        if k in allowed:
            filtered_args[k] = v
        else:
            dropped.append(k)

    if dropped:
        print(
            f"⚠️  Warning: The following parameters were not recognised by '{model_name}' and will be ignored:"
        )
        for param in dropped:
            print(f"   - {param}: {model_args[param]}")

    return model_cls(**filtered_args) 
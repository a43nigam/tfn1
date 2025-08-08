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
from model.wrappers import create_revin_wrapper, create_parn_wrapper

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

    # Build the base model
    base_model = model_cls(**filtered_args)
    
    # ---- apply normalization wrappers if specified ------------------------
    normalization_config = model_cfg.get("normalization", {})
    normalization_type = normalization_config.get("type", None)
    
    if normalization_type is not None:
        # Determine number of features for normalization
        num_features = normalization_config.get("num_features")
        if num_features is None:
            # Try to infer from model configuration
            if "input_dim" in model_cfg:
                num_features = model_cfg["input_dim"]
            elif "embed_dim" in model_cfg:
                num_features = model_cfg["embed_dim"]
            else:
                raise ValueError(
                    f"Cannot determine num_features for {normalization_type} normalization. "
                    f"Please specify 'num_features' in the normalization config."
                )
        
        if normalization_type.lower() == "revin":
            print(f"🔧 Wrapping model with RevIN normalization (num_features={num_features})")
            return create_revin_wrapper(base_model, num_features)
        
        elif normalization_type.lower() == "parn":
            mode = normalization_config.get("mode", "location")
            print(f"🔧 Wrapping model with PARN normalization (mode={mode})")
            return create_parn_wrapper(base_model, num_features, mode)
        
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}. Use 'revin' or 'parn'")
    
    return base_model 
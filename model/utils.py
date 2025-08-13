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
from model.wrappers import create_revin_wrapper, PARNModel

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

    # ---- apply normalization wrappers if specified ------------------------
    # Check for normalization_strategy in data_cfg (consistent with train.py)
    normalization_strategy = None
    if data_cfg is not None:
        normalization_strategy = data_cfg.get("normalization_strategy")
    
    if normalization_strategy is not None:
        # Determine number of features for normalization
        num_features = None
        if "input_dim" in filtered_args:
            num_features = filtered_args["input_dim"]
        elif "embed_dim" in filtered_args:
            num_features = filtered_args["embed_dim"]
        else:
            raise ValueError(
                f"Cannot determine num_features for {normalization_strategy} normalization. "
                f"Please specify 'input_dim' or 'embed_dim' in the model configuration."
            )
        
        if normalization_strategy.lower() == "instance":
            # Build the base model for RevIN wrapping
            base_model = model_cls(**filtered_args)
            print(f"🔧 Wrapping model with RevIN normalization (num_features={num_features})")
            return create_revin_wrapper(base_model, num_features)
        
        elif normalization_strategy.lower() == "parn":
            # --- START FIX: ADD PARN WRAPPER LOGIC ---
            parn_mode = data_cfg.get("parn_mode", "location")
            print(f"🔧 Wrapping model '{model_name}' with PARN wrapper.")
            
            # 1. Build the base model first (with all its parameters)
            base_model = model_cls(**filtered_args)
            
            # 2. Build the PARNModel wrapper around the base model
            return PARNModel(
                base_model=base_model,
                num_features=filtered_args['input_dim'],
                mode=parn_mode
            )
            # --- END FIX ---
        
        else:
            raise ValueError(f"Unknown normalization strategy: {normalization_strategy}. Use 'instance' or 'parn'")
    
    # Build the base model if no normalization wrapper is specified
    base_model = model_cls(**filtered_args)
    return base_model 
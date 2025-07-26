"""
Model Registry for TFN Unified Training System

This module provides a centralized registry of all available models with their
parameters, evolution types, and compatibility information.
"""

from typing import Dict, List, Any, Optional
# Legacy classifier/regressor aliases were removed; map to core TFN
from .tfn_unified import TFN as TFNClassifier, TFN as TFNRegressor, UnifiedTFN as _Alias
# Placeholder enhanced alias (points to core TFN or future enhanced)
EnhancedTFNClassifier = TFNClassifier

# Time-series regressors map to the same TFN for now (kept for registry compatibility)
# TFNTimeSeriesRegressor = TFNRegressor
# TFNMultiStepRegressor = TFNRegressor
# TFNSequenceRegressor = TFNRegressor
# Time-series regressors can still map to UnifiedTFN if needed; keep imports minimal
# Deprecated/legacy 2-D implementation replaced by ImageTFN but registry keeps entry for compat
# Enhanced TFN model
from .tfn_enhanced import EnhancedTFNModel
from .tfn_pytorch import ImageTFN as TFNClassifier2D  # alias for 2D image model
from .baselines import TransformerBaseline, PerformerBaseline, LSTMBaseline, CNNBaseline
from .seq_baselines import TFNSeqModel, SimpleTransformerSeqModel, SimplePerformerSeqModel

# Model Registry with all parameters and evolution types
MODEL_REGISTRY = {
    # ============================================================================
    # BASE TFN MODELS (FieldEvolver evolution types)
    # ============================================================================
    
    'tfn_classifier': {
        'class': TFNClassifier,
        'task_type': 'classification',
        'evolution_types': ['cnn', 'pde'],
        'components': ['field_projection', 'field_evolution', 'field_sampling'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes', 'kernel_type', 'evolution_type'],
        'optional_params': ['num_layers', 'grid_size', 'time_steps', 'dropout'],
        'defaults': {
            'num_layers': 2,
            'kernel_type': 'rbf',
            'evolution_type': 'cnn',
            'grid_size': 100,
            'time_steps': 3,
            'dropout': 0.1
        }
    },
    
    'tfn_regressor': {
        'class': TFNRegressor,
        'task_type': 'regression',
        'evolution_types': ['cnn', 'pde'],
        'components': ['field_projection', 'field_evolution', 'field_sampling'],
        'required_params': ['input_dim', 'embed_dim', 'output_dim', 'kernel_type', 'evolution_type'],
        'optional_params': ['num_layers', 'grid_size', 'time_steps', 'dropout'],
        'defaults': {
            'num_layers': 2,
            'kernel_type': 'rbf',
            'evolution_type': 'cnn',
            'grid_size': 100,
            'time_steps': 3,
            'dropout': 0.1
        }
    },
    
    'tfn_language_model': {
        'class': TFNSeqModel,
        'task_type': 'language_modeling',
        'evolution_types': ['cnn', 'pde'],
        'components': ['field_projection', 'field_evolution', 'field_sampling'],
        'required_params': ['vocab_size', 'embed_dim', 'kernel_type', 'evolution_type'],
        'optional_params': ['seq_len', 'grid_size', 'time_steps', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'grid_size': 256,
            'kernel_type': 'rbf',
            'evolution_type': 'cnn',
            'time_steps': 3,
            'dropout': 0.1
        }
    },
    
    'tfn_vision': {
        'class': TFNClassifier2D,
        'task_type': 'vision',
        'evolution_types': ['cnn', 'pde'],  # 2D uses different evolution
        'components': ['field_projection', 'field_evolution', 'field_sampling'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes'],
        'optional_params': ['num_layers', 'num_evolution_steps', 'field_dim', 'grid_height', 'grid_width',
                          'use_dynamic_positions', 'learnable_sigma', 'learnable_out_sigma', 'out_sigma_scale',
                          'field_dropout', 'use_global_context', 'dropout', 'multiscale', 'kernel_mix',
                          'kernel_mix_scale'],
        'defaults': {
            'num_layers': 4,
            'num_evolution_steps': 5,
            'field_dim': 64,
            'grid_height': 32,
            'grid_width': 32,
            'use_dynamic_positions': False,
            'learnable_sigma': True,
            'learnable_out_sigma': False,
            'out_sigma_scale': 2.0,
            'field_dropout': 0.0,
            'use_global_context': False,
            'dropout': 0.1,
            'multiscale': False,
            'kernel_mix': False,
            'kernel_mix_scale': 2.0
        }
    },
    
    # ============================================================================
    # ENHANCED TFN MODELS (DynamicFieldPropagator evolution types)
    # ============================================================================
    
    'enhanced_tfn_classifier': {
        'class': EnhancedTFNModel,
        'task_type': 'classification',
        'evolution_types': ['diffusion', 'wave', 'schrodinger', 'cnn'],
        'components': ['field_projection', 'field_interference', 'field_propagation', 'field_operators', 'field_evolution', 'field_sampling'],
        'required_params': ['vocab_size', 'embed_dim', 'kernel_type', 'interference_type', 'propagator_type', 'operator_type'],
        'optional_params': ['num_layers', 'evolution_type', 'grid_size', 'num_heads', 'dropout'],
        'physics_params': ['use_physics_constraints', 'constraint_weight'],
        'defaults': {
            'num_layers': 2,
            'kernel_type': 'rbf',
            'evolution_type': 'diffusion',
            'interference_type': 'standard',
            'propagator_type': 'standard',
            'operator_type': 'standard',
            'grid_size': 100,
            'num_heads': 8,
            'dropout': 0.1,
            'use_physics_constraints': False,
            'constraint_weight': 0.1
        }
    },
    
    'enhanced_tfn_language_model': {
        'class': EnhancedTFNModel,
        'task_type': 'language_modeling',
        'evolution_types': ['diffusion', 'wave', 'schrodinger', 'cnn'],
        'components': ['field_projection', 'field_interference', 'field_propagation', 'field_operators', 'field_evolution', 'field_sampling'],
        'required_params': ['vocab_size', 'embed_dim', 'num_layers', 'kernel_type', 'interference_type', 'propagator_type', 'operator_type'],
        'optional_params': ['pos_dim', 'evolution_type', 'grid_size', 'num_heads', 'dropout', 'max_seq_len'],
        'physics_params': ['use_physics_constraints', 'constraint_weight'],
        'defaults': {
            'pos_dim': 1,
            'kernel_type': 'rbf',
            'evolution_type': 'diffusion',
            'interference_type': 'standard',
            'propagator_type': 'standard',
            'operator_type': 'standard',
            'grid_size': 100,
            'num_heads': 8,
            'dropout': 0.1,
            'max_seq_len': 512,
            'use_physics_constraints': False,
            'constraint_weight': 0.1
        }
    },
    
    # ============================================================================
    # BASELINE MODELS (no evolution types)
    # ============================================================================
    
    'transformer_classifier': {
        'class': TransformerBaseline,
        'task_type': 'classification',
        'evolution_types': [],
        'components': ['transformer_encoder', 'classification_head'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes'],
        'optional_params': ['seq_len', 'num_layers', 'num_heads', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    },
    
    'performer_classifier': {
        'class': PerformerBaseline,
        'task_type': 'classification',
        'evolution_types': [],
        'components': ['performer_encoder', 'classification_head'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes'],
        'optional_params': ['seq_len', 'num_layers', 'proj_dim', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'proj_dim': 64,
            'dropout': 0.1
        }
    },
    
    'lstm_classifier': {
        'class': LSTMBaseline,
        'task_type': 'classification',
        'evolution_types': [],
        'components': ['lstm_encoder', 'classification_head'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes'],
        'optional_params': ['hidden_dim', 'num_layers', 'dropout', 'bidirectional'],
        'defaults': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': True
        }
    },
    
    'cnn_classifier': {
        'class': CNNBaseline,
        'task_type': 'classification',
        'evolution_types': [],
        'components': ['cnn_encoder', 'classification_head'],
        'required_params': ['vocab_size', 'embed_dim', 'num_classes'],
        'optional_params': ['num_filters', 'filter_sizes', 'dropout'],
        'defaults': {
            'num_filters': 128,
            'filter_sizes': [3, 4, 5],
            'dropout': 0.1
        }
    },
    
    'transformer_regressor': {
        'class': TransformerBaseline,
        'task_type': 'regression',
        'evolution_types': [],
        'components': ['transformer_encoder', 'regression_head'],
        'required_params': ['input_dim', 'embed_dim', 'output_dim', 'output_len'],
        'optional_params': ['seq_len', 'num_layers', 'num_heads', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    },
    
    'performer_regressor': {
        'class': PerformerBaseline,
        'task_type': 'regression',
        'evolution_types': [],
        'components': ['performer_encoder', 'regression_head'],
        'required_params': ['input_dim', 'embed_dim', 'output_dim', 'output_len'],
        'optional_params': ['seq_len', 'num_layers', 'proj_dim', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'proj_dim': 64,
            'dropout': 0.1
        }
    },
    
    'lstm_regressor': {
        'class': LSTMBaseline,
        'task_type': 'regression',
        'evolution_types': [],
        'components': ['lstm_encoder', 'regression_head'],
        'required_params': ['input_dim', 'embed_dim', 'output_dim', 'output_len'],
        'optional_params': ['hidden_dim', 'num_layers', 'dropout', 'bidirectional'],
        'defaults': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': True
        }
    },
    
    'cnn_regressor': {
        'class': CNNBaseline,
        'task_type': 'regression',
        'evolution_types': [],
        'components': ['cnn_encoder', 'regression_head'],
        'required_params': ['input_dim', 'embed_dim', 'output_dim', 'output_len'],
        'optional_params': ['num_filters', 'filter_sizes', 'dropout'],
        'defaults': {
            'num_filters': 128,
            'filter_sizes': [3, 4, 5],
            'dropout': 0.1
        }
    },
    
    'transformer_language_model': {
        'class': SimpleTransformerSeqModel,
        'task_type': 'language_modeling',
        'evolution_types': [],
        'components': ['transformer_encoder', 'language_model_head'],
        'required_params': ['vocab_size', 'embed_dim'],
        'optional_params': ['seq_len', 'num_layers', 'num_heads', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    },
    
    'performer_language_model': {
        'class': SimplePerformerSeqModel,
        'task_type': 'language_modeling',
        'evolution_types': [],
        'components': ['performer_encoder', 'language_model_head'],
        'required_params': ['vocab_size', 'embed_dim'],
        'optional_params': ['seq_len', 'num_layers', 'proj_dim', 'dropout'],
        'defaults': {
            'seq_len': 512,
            'num_layers': 2,
            'proj_dim': 64,
            'dropout': 0.1
        }
    }
}

# Task compatibility matrix
TASK_COMPATIBILITY = {
    'classification': {
        'models': ['tfn_classifier', 'enhanced_tfn_classifier', 'transformer_classifier', 
                  'performer_classifier', 'lstm_classifier', 'cnn_classifier'],
        'datasets': ['sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'cola', 'wnli', 'arxiv']
    },
    'regression': {
        'models': ['tfn_regressor', 'transformer_regressor', 'performer_regressor', 
                  'lstm_regressor', 'cnn_regressor'],
        'datasets': ['stsb']
    },
    'time_series': {
        'models': ['tfn_timeseries_regressor', 'tfn_multistep_regressor', 'tfn_sequence_regressor',
                  'transformer_regressor', 'performer_regressor', 'lstm_regressor', 'cnn_regressor'],
        'datasets': ['electricity', 'jena', 'jena_multi']
    },
    'language_modeling': {
        'models': ['tfn_language_model', 'enhanced_tfn_language_model', 'transformer_language_model', 
                  'performer_language_model'],
        'datasets': ['pg19', 'long_text']
    },
    'vision': {
        'models': ['tfn_vision'],
        'datasets': ['cifar10']
    },
    'ner': {
        'models': [],  # No NER models implemented yet
        'datasets': ['conll2003']
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]

def get_evolution_types(model_name: str) -> List[str]:
    """Get available evolution types for a model."""
    config = get_model_config(model_name)
    return config.get('evolution_types', [])

def get_task_compatibility(task: str) -> Dict[str, List[str]]:
    """Get compatible models and datasets for a task."""
    if task not in TASK_COMPATIBILITY:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(TASK_COMPATIBILITY.keys())}")
    return TASK_COMPATIBILITY[task]

def validate_model_task_compatibility(model_name: str, task: str) -> bool:
    """Validate that a model is compatible with a task."""
    task_config = get_task_compatibility(task)
    return model_name in task_config['models']

def get_required_params(model_name: str) -> List[str]:
    """Get required parameters for a model."""
    config = get_model_config(model_name)
    return config.get('required_params', [])

def get_optional_params(model_name: str) -> List[str]:
    """Get optional parameters for a model."""
    config = get_model_config(model_name)
    return config.get('optional_params', [])

def get_physics_params(model_name: str) -> List[str]:
    """Get physics parameters for a model."""
    config = get_model_config(model_name)
    return config.get('physics_params', [])

def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """Get default parameters for a model."""
    config = get_model_config(model_name)
    return config.get('defaults', {}) 

# ---------------------------------------------------------------------------
# Kernel / evolution compatibility helper
# ---------------------------------------------------------------------------

# Mapping evolution_type → allowed kernels
_ALLOWED_KERNELS = {
    "cnn": {"rbf", "compact"},
    "pde": {"rbf", "fourier"},
    "diffusion": {"rbf", "fourier"},
    "wave": {"rbf", "fourier"},
    "schrodinger": {"fourier"},
}


def validate_kernel_evolution(kernel_type: str, evolution_type: str) -> None:
    """Raise ValueError if *kernel_type* is incompatible with *evolution_type*.

    The mapping is conservative – we only allow pairs that have been
    implemented and numerically verified. If you extend kernels or evolutions
    update `_ALLOWED_KERNELS` and add unit tests.
    """
    allowed = _ALLOWED_KERNELS.get(evolution_type, set())
    if kernel_type not in allowed:
        raise ValueError(
            f"kernel_type='{kernel_type}' is not supported with evolution_type='{evolution_type}'. "
            f"Allowed: {sorted(allowed)}"
        ) 
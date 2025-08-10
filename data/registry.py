"""
Data Loader Registry for TFN Unified Training System

This module provides a centralized registry of all available data loaders with their
parameters, task types, and compatibility information.
"""

from typing import Dict, List, Any, Optional, Callable, Type, Tuple
from torch.utils.data import Dataset
from data.timeseries_loader import ETTDataset
from data.nlp_loader import NLPDataset
from data.jena_loader import JenaClimateDataset
from data.stock_loader import StockMarketDataset
from data.glue_loader import GLUEDataset
from data.arxiv_loader import ArxivDataset
from data.pg19_loader import PG19Dataset
from data.wikitext_loader import WikiTextDataset
from data.pde_loader import PDEDataset, DarcyFlowDataset, load_pde_data
from data_pipeline import SyntheticCopyDataset
from data.synthetic_task_loader import SyntheticTaskDataset
from data.split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC
import torch

# Create a BurgersDataset class for the registry
class BurgersDataset(PDEDataset):
    """Burgers' Equation dataset for 1D shock wave modeling."""
    
    def __init__(
        self,
        initial_conditions: torch.Tensor,
        solutions: torch.Tensor,
        grid: torch.Tensor,
        split: str = 'train',
        target_timestep: int = 10,
        normalize: bool = True,
        normalization_strategy: str = 'global',
        scaler: Optional[Any] = None
    ) -> None:
        super().__init__(
            initial_conditions=initial_conditions,
            solutions=solutions,
            grid=grid,
            split=split,
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy,
            scaler=scaler
        )
    
    @staticmethod
    def get_splits(
        file_path: str,
        target_timestep: int = 10,
        split_frac: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        normalization_strategy: str = 'global'
    ) -> Tuple['BurgersDataset', 'BurgersDataset', 'BurgersDataset']:
        """Factory method for Burgers' equation datasets."""
        # Load data once
        initial_conditions, solutions, grid = load_pde_data(file_path, 'a', 'u', 'x')
        
        # Determine split sizes
        if split_frac is None:
            split_frac = DEFAULT_SPLIT_FRAC
        
        n_samples = initial_conditions.shape[0]
        n_train, n_val, n_test = get_split_sizes(n_samples, split_frac)
        
        # Create splits
        train_initial = initial_conditions[:n_train]
        train_solutions = solutions[:n_train]
        
        val_initial = initial_conditions[n_train:n_train + n_val]
        val_solutions = solutions[n_train:n_train + n_val]
        
        test_initial = initial_conditions[n_train + n_val:]
        test_solutions = solutions[n_train + n_val:]
        
        # Create dataset instances
        train_ds = BurgersDataset(
            initial_conditions=train_initial,
            solutions=train_solutions,
            grid=grid,
            split='train',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        val_ds = BurgersDataset(
            initial_conditions=val_initial,
            solutions=val_solutions,
            grid=grid,
            split='val',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        test_ds = BurgersDataset(
            initial_conditions=test_initial,
            solutions=test_solutions,
            grid=grid,
            split='test',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        return train_ds, val_ds, test_ds

# Data Loader Registry with all parameters and task types
DATASET_REGISTRY = {
    # ============================================================================
    # SYNTHETIC DATASETS
    # ============================================================================
    
    'synthetic': {
        'class': SyntheticCopyDataset,
        'task_type': 'copy',  # Can be 'copy' or 'classification'
        'required_params': ['dataset_size', 'seq_len', 'vocab_size'],
        'optional_params': ['pad_idx', 'task', 'num_classes'],
        'defaults': {
            'dataset_size': 1000,
            'seq_len': 50,
            'vocab_size': 20,
            'pad_idx': 0,
            'task': 'copy',
            'num_classes': 2
        },
        'factory_method': None,  # Direct instantiation
        'split_method': None,    # No splits for synthetic
        'description': 'Synthetic dataset for testing copy and classification tasks'
    },
    
    'heat_equation': {
        'class': SyntheticTaskDataset,
        'task_type': 'regression',
        'required_params': ['file_path'],
        'optional_params': ['split_ratios', 'seed'],
        'defaults': {
            'split_ratios': None,
            'seed': 42
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Synthetic 1D heat equation simulation for testing PDE learning and spatial-temporal evolution'
    },

    'delayed_copy': {
        'class': SyntheticTaskDataset,
        'task_type': 'language_modeling',  # Changed from 'classification' - token-level prediction task
        'required_params': ['file_path'],
        'optional_params': ['split_ratios', 'seed'],
        'defaults': {
            'split_ratios': None,
            'seed': 42
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Synthetic delayed copy task for testing long-range dependencies and memory retention'
    },

    'irregular_sampling': {
        'class': SyntheticTaskDataset,
        'task_type': 'regression',
        'required_params': ['file_path'],
        'optional_params': ['split_ratios', 'seed'],
        'defaults': {
            'split_ratios': None,
            'seed': 42
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Synthetic task with irregular timestamps to test continuous position handling and temporal interpolation'
    },
    
    # ============================================================================
    # TIMESERIES DATASETS
    # ============================================================================
    
    'ett': {
        'class': ETTDataset,
        'task_type': 'regression',
        'required_params': ['csv_path', 'input_len', 'output_len'],
        'optional_params': ['normalization_strategy', 'instance_normalize'],
        'defaults': {
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24,
            'normalization_strategy': 'global',
            'instance_normalize': False
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Electricity Transformer Temperature dataset for time series forecasting'
    },
    
    'jena': {
        'class': JenaClimateDataset,
        'task_type': 'regression',
        'required_params': ['csv_path', 'input_len', 'output_len'],
        'optional_params': [],
        'defaults': {
            'csv_path': 'data/jena_climate_2009_2016.csv',
            'input_len': 96,
            'output_len': 24
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Jena Climate dataset for time series forecasting'
    },
    
    'stock': {
        'class': StockMarketDataset,
        'task_type': 'regression',
        'required_params': ['csv_path', 'input_len', 'output_len'],
        'optional_params': ['ticker'],
        'defaults': {
            'csv_path': 'data/all_stocks_5yr.csv',
            'input_len': 60,
            'output_len': 5,
            'ticker': None
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Stock market dataset for time series forecasting'
    },
    
    # ============================================================================
    # NLP DATASETS
    # ============================================================================
    
    'glue': {
        'class': GLUEDataset,
        'task_type': 'classification',
        'required_params': ['task'],
        'optional_params': ['max_length', 'tokenizer_name'],
        'defaults': {
            'task': 'sst2',
            'max_length': 512,
            'tokenizer_name': 'bert-base-uncased'
        },
        'factory_method': None,  # Direct instantiation
        'split_method': None,    # Uses split parameter
        'description': 'GLUE benchmark datasets for NLP classification tasks'
    },
    
    'arxiv': {
        'class': ArxivDataset,
        'task_type': 'classification',
        'required_params': ['csv_path'],
        'optional_params': ['max_length', 'tokenizer_name'],
        'defaults': {
            'csv_path': 'data/arxiv_sample.csv',
            'max_length': 512,
            'tokenizer_name': 'bert-base-uncased'
        },
        'factory_method': None,  # Direct instantiation
        'split_method': None,    # No splits for arxiv
        'description': 'ArXiv paper classification dataset'
    },
    
    # ============================================================================
    # LANGUAGE MODELING DATASETS
    # ============================================================================
    
    'pg19': {
        'class': PG19Dataset,
        'task_type': 'language_modeling',
        'required_params': ['data_dir'],
        'optional_params': ['max_length', 'tokenizer_name'],
        'defaults': {
            'data_dir': 'data/pg19',
            'max_length': 512,
            'tokenizer_name': 'gpt2'
        },
        'factory_method': None,  # Direct instantiation
        'split_method': None,    # Uses split parameter
        'description': 'Project Gutenberg dataset for language modeling'
    },
    
    'wikitext': {
        'class': WikiTextDataset,
        'task_type': 'language_modeling',
        'required_params': ['wikitext_dataset_name'],
        'optional_params': ['tokenizer_name', 'max_length', 'text_col', 'use_streaming', 'max_samples'],
        'defaults': {
            'wikitext_dataset_name': 'wikitext-2-raw-v1',
            'tokenizer_name': 'gpt2',
            'max_length': 512,
            'text_col': 'text',
            'use_streaming': False,
            'max_samples': None
        },
        'factory_method': None,  # Direct instantiation
        'split_method': None,    # Uses split parameter
        'description': 'WikiText dataset for language modeling'
    },
    
    # ============================================================================
    # PDE DATASETS
    # ============================================================================
    
    'pde': {
        'class': PDEDataset,
        'task_type': 'regression',
        'required_params': ['file_path'],
        'optional_params': ['initial_condition_key', 'solution_key', 'grid_key', 'input_timesteps', 
                          'target_timestep', 'split_frac', 'normalize', 'normalization_strategy'],
        'defaults': {
            'file_path': 'data/pde_data.mat',
            'initial_condition_key': 'a',
            'solution_key': 'u',
            'grid_key': 'x',
            'input_timesteps': 1,
            'target_timestep': 10,
            'normalize': True,
            'normalization_strategy': 'global'
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Generic PDE dataset for physics-informed neural networks'
    },
    
    'burgers': {
        'class': BurgersDataset,
        'task_type': 'regression',
        'required_params': ['file_path'],
        'optional_params': ['target_timestep', 'split_frac', 'normalize', 'normalization_strategy'],
        'defaults': {
            'file_path': 'data/burgers_data.mat',
            'target_timestep': 10,
            'normalize': True,
            'normalization_strategy': 'global'
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Burgers\' Equation dataset for shock wave modeling'
    },
    
    'darcy': {
        'class': DarcyFlowDataset,
        'task_type': 'regression',
        'required_params': ['file_path'],
        'optional_params': ['target_timestep', 'split_frac', 'normalize', 'normalization_strategy'],
        'defaults': {
            'file_path': 'data/darcy_data.mat',
            'target_timestep': 0,  # Darcy flow is typically steady-state
            'normalize': True,
            'normalization_strategy': 'global'
        },
        'factory_method': 'get_splits',
        'split_method': 'get_splits',
        'description': 'Darcy Flow dataset for porous media fluid flow modeling'
    }
}

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name]

def get_task_compatibility(task: str) -> Dict[str, List[str]]:
    """Get all datasets compatible with a specific task."""
    compatible_datasets = []
    for dataset_name, config in DATASET_REGISTRY.items():
        if config['task_type'] == task:
            compatible_datasets.append(dataset_name)
    return {task: compatible_datasets}

def get_required_params(dataset_name: str) -> List[str]:
    """Get required parameters for a dataset."""
    config = get_dataset_config(dataset_name)
    return config['required_params']

def get_optional_params(dataset_name: str) -> List[str]:
    """Get optional parameters for a dataset."""
    config = get_dataset_config(dataset_name)
    return config['optional_params']

def get_dataset_defaults(dataset_name: str) -> Dict[str, Any]:
    """Get default parameters for a dataset."""
    config = get_dataset_config(dataset_name)
    return config['defaults']

def validate_dataset_task_compatibility(dataset_name: str, task: str) -> bool:
    """Validate if a dataset is compatible with a task."""
    config = get_dataset_config(dataset_name)
    return config['task_type'] == task

def list_available_datasets() -> List[str]:
    """List all available datasets."""
    return list(DATASET_REGISTRY.keys())

def list_datasets_by_task(task: str) -> List[str]:
    """List all datasets for a specific task."""
    return get_task_compatibility(task)[task]

def get_dataset_description(dataset_name: str) -> str:
    """Get description for a dataset."""
    config = get_dataset_config(dataset_name)
    return config['description']

def register_dataset(
    name: str,
    dataset_class: Type[Dataset],
    task_type: str,
    required_params: List[str],
    optional_params: List[str] = None,
    defaults: Dict[str, Any] = None,
    factory_method: Optional[str] = None,
    split_method: Optional[str] = None,
    description: str = ""
) -> None:
    """
    Register a new dataset in the registry.
    
    Args:
        name: Dataset name
        dataset_class: Dataset class
        task_type: Task type (regression, classification, language_modeling, etc.)
        required_params: List of required parameters
        optional_params: List of optional parameters
        defaults: Default parameter values
        factory_method: Method name for creating dataset (if different from __init__)
        split_method: Method name for getting splits (if applicable)
        description: Dataset description
    """
    DATASET_REGISTRY[name] = {
        'class': dataset_class,
        'task_type': task_type,
        'required_params': required_params,
        'optional_params': optional_params or [],
        'defaults': defaults or {},
        'factory_method': factory_method,
        'split_method': split_method,
        'description': description
    }

def create_dataset(dataset_name: str, config: Dict[str, Any], split: str = 'train') -> Dataset:
    """
    Create a dataset instance using the registry.
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        Dataset instance
    """
    dataset_config = get_dataset_config(dataset_name)
    dataset_class = dataset_config['class']
    
    # Get parameters from config
    data_cfg = config.get('data', {})
    
    # Merge with defaults
    params = dataset_config['defaults'].copy()
    params.update(data_cfg)
    
    # Handle split mapping for datasets that use different naming conventions
    split_mapping = {
        'val': 'validation',  # WikiText uses 'validation' instead of 'val'
        'validation': 'validation',
        'train': 'train',
        'test': 'test'
    }
    mapped_split = split_mapping.get(split, split)
    
    # Check if dataset uses get_splits method
    if dataset_config['split_method'] == 'get_splits':
        # For datasets with get_splits method
        required_params = dataset_config['required_params']
        call_params = {param: params[param] for param in required_params if param in params}
        
        # Add optional parameters
        for param in dataset_config['optional_params']:
            if param in params:
                call_params[param] = params[param]
        
        # Call get_splits method
        train_ds, val_ds, test_ds = dataset_class.get_splits(**call_params)
        
        if split == 'train':
            return train_ds
        elif split == 'val':
            return val_ds
        elif split == 'test':
            return test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
    
    else:
        # For datasets with direct instantiation
        # Add split parameter if the dataset supports it
        if 'split' in dataset_class.__init__.__code__.co_varnames:
            params['split'] = mapped_split
        
        # Remove parameters that don't exist in the class constructor
        valid_params = {}
        for key, value in params.items():
            if key in dataset_class.__init__.__code__.co_varnames:
                valid_params[key] = value
        
        return dataset_class(**valid_params) 
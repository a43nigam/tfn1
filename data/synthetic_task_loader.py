"""
Synthetic Task Data Loader for TFN

This module provides a unified data loader for pre-generated synthetic tasks.
It handles train/val/test splits and standardizes data formats for compatibility
with the TFN training system.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Any, Optional
from . import resolve_data_path
import numpy as np


class SyntheticTaskDataset(Dataset):
    """
    A generic Dataset for loading pre-generated synthetic tasks.
    
    It loads a .pt file containing tensors and handles train/val/test splits.
    The dataset automatically adapts to different synthetic task formats:
    - Heat equation: initial_conditions + solutions
    - Delayed copy: inputs + targets (discrete tokens)  
    - Irregular sampling: inputs + targets + positions
    """
    
    def __init__(
        self, 
        file_path: str, 
        split: str = 'train',
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42
    ):
        """
        Initialize the synthetic task dataset.
        
        Args:
            file_path: Path to the .pt file containing the dataset
            split: Dataset split ('train', 'val', 'test')
            split_ratios: Custom split ratios (default: 80/10/10)
            seed: Random seed for reproducible splits
        """
        self.file_path = resolve_data_path(file_path)
        self.split = split
        self.seed = seed
        
        # Default split ratios
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        self.split_ratios = split_ratios
        
        # Load data
        self.data = torch.load(self.file_path, map_location='cpu')
        self.metadata = self.data.get('metadata', {})
        
        # Determine dataset type and prepare data
        self._prepare_data()
        
        # Create split indices
        self._create_splits()
    
    def _prepare_data(self) -> None:
        """Prepare and standardize the data format based on the dataset type."""
        
        # Detect dataset type based on keys
        data_keys = set(self.data.keys()) - {'metadata'}
        
        if 'initial_conditions' in data_keys and 'solutions' in data_keys:
            # Heat equation dataset
            self.dataset_type = 'heat_equation'
            
            # --- FIX: Reshape data for TFN compatibility ---
            # Original data shapes:
            # initial_conditions: [n_samples, grid_points]
            # solutions: [n_samples, seq_len, grid_points]
            
            # 1. Reshape the input to [n_samples, grid_points, 1]
            # The spatial grid becomes the sequence dimension
            self.inputs = self.data['initial_conditions'].unsqueeze(-1)  # [n_samples, grid_points, 1]

            # 2. Select a single future timestep as the target
            # This simplifies the task to a standard sequence-to-sequence problem
            target_timestep = self.data['solutions'].shape[1] - 1  # Use the last timestep
            self.targets = self.data['solutions'][:, target_timestep, :].unsqueeze(-1)  # [n_samples, grid_points, 1]
            
            # 3. Store the grid to be used as positions
            if 'grid' in self.data:
                self.grid = self.data['grid']
            else:
                grid_points = self.inputs.shape[1]
                self.grid = torch.linspace(0, 1, grid_points)
            
            # 4. Set positions to None since we'll generate them per-sample in __getitem__
            self.positions = None
            # --- FIX ENDS HERE ---
                
        elif 'inputs' in data_keys and 'targets' in data_keys and 'positions' in data_keys:
            # Irregular sampling dataset
            self.dataset_type = 'irregular_sampling'
            self.inputs = self.data['inputs']      # [n_samples, n_points]
            self.targets = self.data['targets']    # [n_samples, n_points]
            self.positions = self.data['positions'] # [n_samples, n_points]
            
            # Add feature dimension: [batch, seq_len, features]
            self.inputs = self.inputs.unsqueeze(-1)   # [batch, seq_len, 1]
            self.targets = self.targets.unsqueeze(-1) # [batch, seq_len, 1]
            
        elif 'inputs' in data_keys and 'targets' in data_keys:
            # Delayed copy dataset (or similar discrete tasks)
            self.dataset_type = 'delayed_copy'
            self.inputs = self.data['inputs']   # [n_samples, seq_len]
            self.targets = self.data['targets'] # [n_samples, seq_len]
            self.positions = None
            
        else:
            raise ValueError(f"Unknown synthetic dataset format. Available keys: {data_keys}")
        
        self.n_samples = self.inputs.shape[0]
    
    def _create_splits(self) -> None:
        """Create train/val/test splits with deterministic shuffling."""
        # Set seed for reproducible splits
        torch.manual_seed(self.seed)
        
        # Create shuffled indices
        indices = torch.randperm(self.n_samples)
        
        # Calculate split sizes
        n_train = int(self.split_ratios['train'] * self.n_samples)
        n_val = int(self.split_ratios['val'] * self.n_samples)
        n_test = self.n_samples - n_train - n_val
        
        # Create split index ranges
        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train', 'val', or 'test'")
    
    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns a dictionary with standardized keys for compatibility with TFN trainer:
        - 'inputs': Input tensor
        - 'targets': Target tensor
        - 'positions': Position/timestamp tensor (if applicable)
        - Additional metadata keys as needed
        """
        # Map to original dataset index
        original_idx = self.indices[idx]
        
        # Build the sample dictionary
        sample = {
            'inputs': self.inputs[original_idx],
            'targets': self.targets[original_idx]
        }
        
        # Add positions if available (for irregular sampling)
        if self.positions is not None:
            sample['positions'] = self.positions[original_idx]
        
        # Add dataset-specific metadata
        if self.dataset_type == 'heat_equation':
            # --- FIX: Ensure positions are included for heat equation ---
            # For heat equation, the spatial grid serves as the positions
            if hasattr(self, 'grid'):
                # The grid is the same for all samples, reshape to [grid_points, 1]
                sample['positions'] = self.grid.unsqueeze(-1)  # Shape: [grid_points, 1]
                sample['grid'] = self.grid  # Keep grid for reference
            # --- FIX ENDS HERE ---
                
        elif self.dataset_type == 'delayed_copy':
            # For delayed copy, add vocab info
            if 'vocab_size' in self.metadata:
                sample['vocab_size'] = torch.tensor(self.metadata['vocab_size'])
            if 'delay' in self.metadata:
                sample['delay'] = torch.tensor(self.metadata['delay'])
        
        return sample
    
    @staticmethod
    def get_splits(
        file_path: str, 
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42
    ) -> Tuple['SyntheticTaskDataset', 'SyntheticTaskDataset', 'SyntheticTaskDataset']:
        """
        Factory method to create train, val, and test splits.
        
        Args:
            file_path: Path to the .pt file containing the dataset
            split_ratios: Custom split ratios (default: 80/10/10)
            seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_ds = SyntheticTaskDataset(file_path, split='train', split_ratios=split_ratios, seed=seed)
        val_ds = SyntheticTaskDataset(file_path, split='val', split_ratios=split_ratios, seed=seed)
        test_ds = SyntheticTaskDataset(file_path, split='test', split_ratios=split_ratios, seed=seed)
        
        return train_ds, val_ds, test_ds
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset."""
        meta = self.metadata.copy()
        meta.update({
            'dataset_type': self.dataset_type,
            'n_samples': self.n_samples,
            'split': self.split,
            'split_size': len(self),
            'input_shape': tuple(self.inputs.shape[1:]),  # Shape without batch dimension
            'target_shape': tuple(self.targets.shape[1:])
        })
        
        if self.positions is not None:
            meta['position_shape'] = tuple(self.positions.shape[1:])
        
        return meta
    
    def get_sample_shapes(self) -> Dict[str, torch.Size]:
        """Get tensor shapes for a single sample (useful for model initialization)."""
        sample = self[0]
        return {key: tensor.shape for key, tensor in sample.items() if isinstance(tensor, torch.Tensor)}


# Convenience functions for direct dataset creation
def load_heat_equation_dataset(
    file_path: str = 'data/synthetic/heat_equation.pt', 
    split: str = 'train'
) -> SyntheticTaskDataset:
    """Load the heat equation dataset directly."""
    return SyntheticTaskDataset(file_path, split=split)


def load_delayed_copy_dataset(
    file_path: str = 'data/synthetic/delayed_copy.pt', 
    split: str = 'train'
) -> SyntheticTaskDataset:
    """Load the delayed copy dataset directly."""
    return SyntheticTaskDataset(file_path, split=split)


def load_irregular_sampling_dataset(
    file_path: str = 'data/synthetic/irregular_sampling.pt', 
    split: str = 'train'
) -> SyntheticTaskDataset:
    """Load the irregular sampling dataset directly."""
    return SyntheticTaskDataset(file_path, split=split) 
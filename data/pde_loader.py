"""
PDE Data Loader for TFN Benchmarking

This module provides data loaders for PDE benchmark datasets commonly used in
Fourier Neural Operator (FNO) and physics-informed neural network research.
Supports datasets like Burgers' Equation, Darcy Flow, and Navier-Stokes.
"""

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from .split_utils import get_split_sizes, DEFAULT_SPLIT_FRAC


def load_pde_data(
    file_path: str,
    initial_condition_key: str = 'a',
    solution_key: str = 'u',
    grid_key: str = 'x',
    grid_points: Optional[int] = None,  # New parameter
    domain: List[float] = [0.0, 1.0]   # New parameter
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess PDE data from a .mat file.
    
    This function loads the data once and handles all the orientation issues,
    making it efficient for creating multiple dataset splits.
    
    Args:
        file_path: Path to the .mat file
        initial_condition_key: Key for initial conditions in .mat file
        solution_key: Key for solutions in .mat file
        grid_key: Key for spatial grid in .mat file
        grid_points: Number of grid points for uniform grid generation (if grid_key not found)
        domain: Domain bounds [min, max] for uniform grid generation (if grid_key not found)
        
    Returns:
        Tuple of (initial_conditions, solutions, grid) as torch tensors
    """
    # Load the .mat file once
    raw_data = loadmat(file_path)
    
    # Extract data
    initial_conditions = raw_data[initial_condition_key]
    solutions = raw_data[solution_key]
    
    # Handle different possible orientations for initial_conditions
    if initial_conditions.ndim == 2:
        # Shape: [n_samples, n_spatial_points] or [n_spatial_points, n_samples]
        # We assume the data is already in [n_samples, n_spatial_points] format
        pass
    else:
        raise ValueError(f"Expected 2D initial conditions, got shape {initial_conditions.shape}")
    
    # Handle different possible orientations for solutions
    if solutions.ndim == 3:
        # Shape: [n_samples, n_spatial_points, n_timesteps] or [n_samples, n_timesteps, n_spatial_points]
        if solutions.shape[1] != initial_conditions.shape[1]:
            # Assume [n_samples, n_timesteps, n_spatial_points] -> transpose to [n_samples, n_spatial_points, n_timesteps]
            solutions = solutions.transpose(0, 2, 1)
    else:
        raise ValueError(f"Expected 3D solutions, got shape {solutions.shape}")
    
    # Validate shapes
    n_samples = initial_conditions.shape[0]
    n_spatial_points = initial_conditions.shape[1]
    
    if solutions.shape[0] != n_samples:
        raise ValueError(f"Sample count mismatch: initial_conditions={n_samples}, solutions={solutions.shape[0]}")
    if solutions.shape[1] != n_spatial_points:
        raise ValueError(f"Spatial points mismatch: initial_conditions={n_spatial_points}, solutions={solutions.shape[1]}")
    
    # --- REVISED GRID HANDLING LOGIC ---
    if grid_key in raw_data:
        # If the key exists, use it as before
        grid_data = raw_data[grid_key]
        
        # Handle grid data
        if grid_data.ndim == 1:
            grid = torch.tensor(grid_data, dtype=torch.float32)
        elif grid_data.ndim == 2:
            if grid_data.shape[0] == 1:  # Shape: [1, n_spatial_points] -> squeeze
                grid = torch.tensor(grid_data.squeeze(), dtype=torch.float32)
            elif grid_data.shape[1] == 1:  # Shape: [n_spatial_points, 1] -> squeeze
                grid = torch.tensor(grid_data.squeeze(), dtype=torch.float32)
            else:  # For 2D grids like Darcy flow: [n_spatial_points, 2]
                grid = torch.tensor(grid_data, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 1D or 2D grid, got shape {grid_data.shape}")
        
        # Ensure grid is 1D for 1D problems or 2D for 2D problems
        if grid.ndim == 2 and grid.shape[1] == 1:
            grid = grid.squeeze(-1)
        
        # Validate grid shape
        if grid.shape[0] != n_spatial_points:
            raise ValueError(f"Grid points mismatch: {n_spatial_points} vs {grid.shape[0]}")
            
    else:
        # If the key is missing, generate the grid programmatically
        if grid_points is None:
            # Infer grid points from the spatial dimension of the data
            grid_points = raw_data[initial_condition_key].shape[1]

        print(f"⚠️  Grid key '{grid_key}' not found in .mat file. Generating a uniform spatial grid with {grid_points} points over the domain {domain}.")
        grid = torch.linspace(domain[0], domain[1], grid_points, dtype=torch.float32)
    # --- END REVISED LOGIC ---
    
    # Convert to torch tensors
    initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
    solutions = torch.tensor(solutions, dtype=torch.float32)
    
    return initial_conditions, solutions, grid


class PDEDataset(Dataset):
    """
    Generic Dataset for PDE benchmarks like Burgers' or Darcy Flow.
    
    This version is optimized for efficiency by accepting pre-loaded data
    instead of loading the .mat file multiple times.
    
    The dataset structure follows FNO conventions:
    - Input: Initial conditions a with shape [n_samples, n_spatial_points]
    - Target: Solutions u at future timestep with shape [n_samples, n_spatial_points]
    - Grid: Spatial coordinates x with shape [n_spatial_points]
    
    Args:
        initial_conditions: Pre-loaded initial conditions tensor [n_samples, n_spatial_points]
        solutions: Pre-loaded solutions tensor [n_samples, n_spatial_points, n_timesteps]
        grid: Pre-loaded spatial grid tensor [n_spatial_points] or [n_spatial_points, 2]
        split: Dataset split ('train', 'val', 'test')
        target_timestep: Target timestep to predict (default: 10)
        normalize: Whether to normalize the data (default: True)
        normalization_strategy: Normalization strategy ('global', 'instance') (default: 'global')
        scaler: Optional scaler object for denormalization
    """
    
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
        super().__init__()
        
        self.split = split
        self.target_timestep = target_timestep
        self.normalize = normalize
        self.normalization_strategy = normalization_strategy
        self.scaler = scaler
        
        # Store the pre-loaded data
        self.initial_conditions = initial_conditions
        self.solutions = solutions
        self.grid = grid
        
        # Validate shapes
        n_samples = self.initial_conditions.shape[0]
        n_spatial_points = self.initial_conditions.shape[1]
        
        if self.solutions.shape[0] != n_samples:
            raise ValueError(f"Mismatch in number of samples: {n_samples} vs {self.solutions.shape[0]}")
        if self.solutions.shape[1] != n_spatial_points:
            raise ValueError(f"Mismatch in spatial points: {n_spatial_points} vs {self.solutions.shape[1]}")
        if self.grid.shape[0] != n_spatial_points:
            raise ValueError(f"Mismatch in grid points: {n_spatial_points} vs {self.grid.shape[0]}")
        
        # Store metadata
        self.metadata = {
            'n_samples': len(self.initial_conditions),
            'n_spatial_points': n_spatial_points,
            'n_timesteps': self.solutions.shape[2] if self.solutions.ndim == 3 else 1,
            'grid_dim': self.grid.ndim,
            'split': split
        }
        
        # Normalize the data if requested
        if normalize:
            self._normalize_data()
    
    def _normalize_data(self) -> None:
        """Normalize the data using the specified strategy."""
        if self.normalization_strategy == 'global':
            # Global normalization across all samples
            self.initial_conditions_mean = torch.mean(self.initial_conditions)
            self.initial_conditions_std = torch.std(self.initial_conditions)
            self.solutions_mean = torch.mean(self.solutions)
            self.solutions_std = torch.std(self.solutions)
            
            # Normalize
            self.initial_conditions = (self.initial_conditions - self.initial_conditions_mean) / self.initial_conditions_std
            self.solutions = (self.solutions - self.solutions_mean) / self.solutions_std
            
        elif self.normalization_strategy == 'instance':
            # Instance normalization (per sample)
            self.initial_conditions_mean = torch.mean(self.initial_conditions, dim=1, keepdim=True)
            self.initial_conditions_std = torch.std(self.initial_conditions, dim=1, keepdim=True)
            self.solutions_mean = torch.mean(self.solutions, dim=1, keepdim=True)
            self.solutions_std = torch.std(self.solutions, dim=1, keepdim=True)
            
            # Normalize
            self.initial_conditions = (self.initial_conditions - self.initial_conditions_mean) / self.initial_conditions_std
            self.solutions = (self.solutions - self.solutions_mean) / self.solutions_std
    
    def __len__(self) -> int:
        return self.initial_conditions.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'inputs': Initial condition [n_spatial_points, input_dim]
            - 'targets': Target solution [n_spatial_points, output_dim]
            - 'positions': Spatial grid coordinates [n_spatial_points, grid_dim]
        """
        # Get initial condition and target solution
        initial_condition = self.initial_conditions[idx]  # [n_spatial_points]
        
        # Handle both time-dependent and steady-state problems
        if self.solutions.shape[2] > 1:
            target_solution = self.solutions[idx, :, self.target_timestep]  # [n_spatial_points]
        else:
            target_solution = self.solutions[idx, :, 0]  # [n_spatial_points]
        
        # Prepare positions (spatial coordinates)
        if self.grid.ndim == 1:
            positions = self.grid.unsqueeze(-1)  # [n_spatial_points, 1]
        else:
            positions = self.grid  # [n_spatial_points, grid_dim]
        
        # Ensure inputs and targets have the right shape for the model
        # Model expects [n_spatial_points, input_dim] and [n_spatial_points, output_dim]
        inputs = initial_condition.unsqueeze(-1)  # [n_spatial_points, 1]
        targets = target_solution.unsqueeze(-1)   # [n_spatial_points, 1]
        
        return {
            'inputs': inputs,
            'targets': targets,
            'positions': positions
        }
    
    @staticmethod
    def get_splits(
        file_path: str,
        initial_condition_key: str = 'a',
        solution_key: str = 'u',
        grid_key: str = 'x',
        target_timestep: int = 10,
        split_frac: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        normalization_strategy: str = 'global',
        grid_points: Optional[int] = None,  # Add this parameter
        domain: List[float] = [0.0, 1.0],   # Add this parameter
        **kwargs  # Add kwargs to handle other potential params
    ) -> Tuple['PDEDataset', 'PDEDataset', 'PDEDataset']:
        """
        Factory method to create train, val, and test splits.
        
        This method loads the data once and creates all three splits efficiently.
        
        Args:
            file_path: Path to the .mat file
            initial_condition_key: Key for initial conditions in .mat file
            solution_key: Key for solutions in .mat file
            grid_key: Key for spatial grid in .mat file
            target_timestep: Target timestep to predict
            split_frac: Custom split fractions
            normalize: Whether to normalize the data
            normalization_strategy: Normalization strategy
            grid_points: Number of grid points for uniform grid generation (if grid_key not found)
            domain: Domain bounds [min, max] for uniform grid generation (if grid_key not found)
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load data once, passing the new grid parameters through
        initial_conditions, solutions, grid = load_pde_data(
            file_path,
            initial_condition_key=kwargs.get('initial_condition_key', initial_condition_key),
            solution_key=kwargs.get('solution_key', solution_key),
            grid_key=kwargs.get('grid_key', grid_key),
            grid_points=grid_points,  # Pass it here
            domain=domain            # Pass it here
        )
        
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
        train_ds = PDEDataset(
            initial_conditions=train_initial,
            solutions=train_solutions,
            grid=grid,
            split='train',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        val_ds = PDEDataset(
            initial_conditions=val_initial,
            solutions=val_solutions,
            grid=grid,
            split='val',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        test_ds = PDEDataset(
            initial_conditions=test_initial,
            solutions=test_solutions,
            grid=grid,
            split='test',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        return train_ds, val_ds, test_ds


class DarcyFlowDataset(PDEDataset):
    """
    Specialized dataset for Darcy Flow.
    
    Darcy flow is a steady-state problem modeling fluid flow through porous media.
    The dataset typically contains:
    - Initial conditions: Permeability fields
    - Solutions: Pressure fields (steady-state)
    - Grid: 2D spatial coordinates
    """
    
    def __init__(
        self,
        initial_conditions: torch.Tensor,
        solutions: torch.Tensor,
        grid: torch.Tensor,
        split: str = 'train',
        target_timestep: int = 0,  # Darcy flow is typically steady-state
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Override to handle 2D grid for Darcy flow."""
        item = super().__getitem__(idx)
        
        # Ensure positions are in the correct shape for 2D data
        if self.grid.ndim == 1:
            # 1D grid - keep as is
            pass
        elif self.grid.ndim == 2:
            # 2D grid - ensure shape is [n_spatial_points, 2]
            n_points = self.grid.shape[0]
            item['positions'] = self.grid.reshape(n_points, -1)
        else:
            # Handle case where grid is already in correct shape
            item['positions'] = self.grid
        
        return item
    
    @staticmethod
    def get_splits(
        file_path: str,
        target_timestep: int = 0,
        split_frac: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        normalization_strategy: str = 'global',
        grid_points: Optional[int] = None,  # Add this parameter
        domain: List[float] = [0.0, 1.0]   # Add this parameter
    ) -> Tuple['DarcyFlowDataset', 'DarcyFlowDataset', 'DarcyFlowDataset']:
        """Factory method for Darcy flow datasets."""
        # Load data once
        initial_conditions, solutions, grid = load_pde_data(
            file_path, 
            'a', 
            'u', 
            'x',
            grid_points=grid_points,  # Pass it here
            domain=domain            # Pass it here
        )
        
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
        train_ds = DarcyFlowDataset(
            initial_conditions=train_initial,
            solutions=train_solutions,
            grid=grid,
            split='train',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        val_ds = DarcyFlowDataset(
            initial_conditions=val_initial,
            solutions=val_solutions,
            grid=grid,
            split='val',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        test_ds = DarcyFlowDataset(
            initial_conditions=test_initial,
            solutions=test_solutions,
            grid=grid,
            split='test',
            target_timestep=target_timestep,
            normalize=normalize,
            normalization_strategy=normalization_strategy
        )
        
        return train_ds, val_ds, test_ds 
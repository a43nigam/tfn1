import torch
import numpy as np
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


def simulate_heat_equation(n_samples: int = 1000, seq_len: int = 200, grid_points: int = 100) -> None:
    """
    Generates data for the 1D Heat Equation using finite-difference method.
    
    The heat equation: ∂u/∂t = α * ∂²u/∂x²
    
    Args:
        n_samples: Number of different initial conditions to simulate
        seq_len: Number of time steps to evolve
        grid_points: Number of spatial grid points
    """
    print(f"Generating Heat Equation data: {n_samples} samples, {seq_len} timesteps, {grid_points} grid points")
    
    # Physical parameters
    alpha = 0.01  # Thermal diffusivity
    dx = 1.0 / (grid_points - 1)  # Spatial step
    dt = 0.4 * dx**2 / alpha  # Time step (stability condition)
    
    # Finite difference coefficient
    r = alpha * dt / dx**2
    
    # Generate random initial conditions
    x = np.linspace(0, 1, grid_points)
    initial_conditions = np.zeros((n_samples, grid_points))
    
    for i in range(n_samples):
        # Create various initial condition types
        if i % 4 == 0:
            # Gaussian bump
            center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.05, 0.15)
            initial_conditions[i] = np.exp(-((x - center) / width)**2)
        elif i % 4 == 1:
            # Two Gaussian bumps
            center1, center2 = np.random.uniform(0.1, 0.4), np.random.uniform(0.6, 0.9)
            width = np.random.uniform(0.05, 0.1)
            initial_conditions[i] = (np.exp(-((x - center1) / width)**2) + 
                                   np.exp(-((x - center2) / width)**2))
        elif i % 4 == 2:
            # Step function
            step_pos = np.random.uniform(0.3, 0.7)
            initial_conditions[i] = (x < step_pos).astype(float)
        else:
            # Random Fourier series
            n_modes = np.random.randint(3, 8)
            for k in range(1, n_modes + 1):
                amp = np.random.uniform(0.1, 0.5) / k
                phase = np.random.uniform(0, 2*np.pi)
                initial_conditions[i] += amp * np.sin(k * np.pi * x + phase)
    
    # Evolve each initial condition
    solutions = np.zeros((n_samples, seq_len, grid_points))
    
    for sample in range(n_samples):
        u = initial_conditions[sample].copy()
        solutions[sample, 0] = u
        
        for t in range(1, seq_len):
            u_new = u.copy()
            # Interior points (explicit finite difference)
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            # Boundary conditions (Dirichlet: u=0 at boundaries)
            u_new[0] = 0
            u_new[-1] = 0
            
            u = u_new
            solutions[sample, t] = u
    
    # Convert to torch tensors
    initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
    solutions = torch.tensor(solutions, dtype=torch.float32)
    
    # Save data
    save_path = os.path.join(os.path.dirname(__file__), 'heat_equation.pt')
    torch.save({
        'initial_conditions': initial_conditions,
        'solutions': solutions,
        'metadata': {
            'n_samples': n_samples,
            'seq_len': seq_len,
            'grid_points': grid_points,
            'alpha': alpha,
            'dx': dx,
            'dt': dt,
            'description': '1D Heat Equation with various initial conditions'
        }
    }, save_path)
    print(f"Heat Equation data saved to {save_path}")
    print(f"Initial conditions shape: {initial_conditions.shape}")
    print(f"Solutions shape: {solutions.shape}")


def create_delayed_copy_task(n_samples: int = 5000, seq_len: int = 10000, delay: int = 9900) -> None:
    """
    Generates data for the long-range delayed copy task.
    
    The task: Copy a short sequence of tokens that appears early in a long sequence.
    This tests the model's ability to maintain long-range dependencies.
    
    Args:
        n_samples: Number of sequences to generate
        seq_len: Total sequence length
        delay: Position where the copy target appears
    """
    print(f"Generating Delayed Copy task: {n_samples} samples, seq_len={seq_len}, delay={delay}")
    
    # Vocabulary parameters
    vocab_size = 10  # Use digits 0-9
    pattern_length = 10  # Length of pattern to copy
    
    inputs = np.zeros((n_samples, seq_len), dtype=np.int64)
    targets = np.zeros((n_samples, seq_len), dtype=np.int64)
    
    for i in range(n_samples):
        # Generate random pattern at the beginning
        pattern = np.random.randint(1, vocab_size, pattern_length)  # Avoid 0 (padding)
        inputs[i, :pattern_length] = pattern
        
        # Fill middle with random noise (but use 0 to distinguish from pattern)
        inputs[i, pattern_length:delay] = 0
        
        # Add delimiter token to signal copy start
        delimiter_token = vocab_size  # Special token
        inputs[i, delay] = delimiter_token
        
        # Target is to copy the pattern after the delimiter
        targets[i, :delay+1] = 0  # No target before copy position
        targets[i, delay+1:delay+1+pattern_length] = pattern
        targets[i, delay+1+pattern_length:] = 0  # No target after pattern
    
    # Convert to torch tensors
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Save data
    save_path = os.path.join(os.path.dirname(__file__), 'delayed_copy.pt')
    torch.save({
        'inputs': inputs,
        'targets': targets,
        'metadata': {
            'n_samples': n_samples,
            'seq_len': seq_len,
            'delay': delay,
            'vocab_size': vocab_size + 1,  # +1 for delimiter
            'pattern_length': pattern_length,
            'description': 'Long-range delayed copy task with random patterns'
        }
    }, save_path)
    print(f"Delayed Copy data saved to {save_path}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")


def create_irregular_sampling_task(n_samples: int = 1000, n_points: int = 200) -> None:
    """
    Generates data with irregularly sampled time points using Poisson process.
    
    This task tests the model's ability to handle non-uniform temporal spacing,
    which is common in real-world time series data.
    
    Args:
        n_samples: Number of time series to generate
        n_points: Number of observation points per series
    """
    print(f"Generating Irregular Sampling task: {n_samples} samples, {n_points} points each")
    
    inputs = np.zeros((n_samples, n_points))
    targets = np.zeros((n_samples, n_points))
    positions = np.zeros((n_samples, n_points))  # Timestamps
    
    for i in range(n_samples):
        # Generate irregular timestamps using Poisson process
        # Exponential inter-arrival times
        inter_arrival_times = np.random.exponential(scale=1.0, size=n_points)
        timestamps = np.cumsum(inter_arrival_times)
        timestamps = timestamps / timestamps[-1]  # Normalize to [0, 1]
        
        positions[i] = timestamps
        
        # Generate underlying signal - mixture of sinusoids with different frequencies
        n_components = np.random.randint(2, 6)
        signal = np.zeros_like(timestamps)
        
        for j in range(n_components):
            frequency = np.random.uniform(0.5, 5.0)
            amplitude = np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            signal += amplitude * np.sin(2 * np.pi * frequency * timestamps + phase)
        
        # Add noise
        noise_level = np.random.uniform(0.05, 0.2)
        signal += noise_level * np.random.randn(n_points)
        
        inputs[i] = signal
        
        # Prediction target: next value (shifted by 1)
        # For the last point, predict extrapolation
        targets[i, :-1] = signal[1:]
        # For last point, use simple linear extrapolation
        if n_points > 1:
            targets[i, -1] = 2 * signal[-1] - signal[-2]
        else:
            targets[i, -1] = signal[-1]
    
    # Convert to torch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    
    # Save data
    save_path = os.path.join(os.path.dirname(__file__), 'irregular_sampling.pt')
    torch.save({
        'inputs': inputs,
        'targets': targets,
        'positions': positions,
        'metadata': {
            'n_samples': n_samples,
            'n_points': n_points,
            'description': 'Irregularly sampled time series with Poisson timestamps'
        }
    }, save_path)
    print(f"Irregular Sampling data saved to {save_path}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Positions shape: {positions.shape}")


def generate_all_datasets() -> None:
    """Generate all synthetic datasets with default parameters."""
    print("="*60)
    print("GENERATING ALL SYNTHETIC DATASETS")
    print("="*60)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # Generate all synthetic datasets
    simulate_heat_equation()
    print("-" * 40)
    
    create_delayed_copy_task()
    print("-" * 40)
    
    create_irregular_sampling_task()
    print("-" * 40)
    
    print("All synthetic datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets() 
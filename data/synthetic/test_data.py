#!/usr/bin/env python3
"""
Test script to verify the integrity of generated synthetic datasets.
"""

import torch
import numpy as np
import os


def test_heat_equation_data():
    """Test the heat equation dataset."""
    print("Testing Heat Equation dataset...")
    
    data_path = os.path.join(os.path.dirname(__file__), 'heat_equation.pt')
    data = torch.load(data_path)
    
    initial_conditions = data['initial_conditions']
    solutions = data['solutions']
    metadata = data['metadata']
    
    print(f"  Initial conditions: {initial_conditions.shape}")
    print(f"  Solutions: {solutions.shape}")
    print(f"  Data range: [{initial_conditions.min():.3f}, {initial_conditions.max():.3f}]")
    
    # Verify that first timestep matches initial conditions
    assert torch.allclose(solutions[:, 0, :], initial_conditions), "First timestep should match initial conditions"
    
    # Check that solutions are finite and not NaN
    assert torch.all(torch.isfinite(solutions)), "All solutions should be finite"
    
    # Check heat equation property: energy should decrease over time (for most initial conditions)
    energy = torch.sum(solutions**2, dim=2)  # L2 norm over space
    energy_decrease = (energy[:, -1] <= energy[:, 0]).float().mean()
    print(f"  Energy decrease ratio: {energy_decrease:.3f} (should be high for heat equation)")
    
    print("  ✓ Heat equation data is valid")


def test_delayed_copy_data():
    """Test the delayed copy task dataset."""
    print("Testing Delayed Copy dataset...")
    
    data_path = os.path.join(os.path.dirname(__file__), 'delayed_copy.pt')
    data = torch.load(data_path)
    
    inputs = data['inputs']
    targets = data['targets']
    metadata = data['metadata']
    
    print(f"  Inputs: {inputs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Vocab size: {metadata['vocab_size']}")
    print(f"  Delay: {metadata['delay']}")
    
    # Check a few examples
    n_check = min(5, inputs.shape[0])
    pattern_length = metadata['pattern_length']
    delay = metadata['delay']
    
    for i in range(n_check):
        # Get the pattern from the beginning
        pattern = inputs[i, :pattern_length]
        
        # Get the target pattern after delay
        target_pattern = targets[i, delay+1:delay+1+pattern_length]
        
        # They should match
        assert torch.equal(pattern, target_pattern), f"Pattern mismatch in sample {i}"
    
    # Check that inputs are within valid range
    assert inputs.min() >= 0 and inputs.max() <= metadata['vocab_size'], "Input tokens out of range"
    
    print("  ✓ Delayed copy data is valid")


def test_irregular_sampling_data():
    """Test the irregular sampling dataset."""
    print("Testing Irregular Sampling dataset...")
    
    data_path = os.path.join(os.path.dirname(__file__), 'irregular_sampling.pt')
    data = torch.load(data_path)
    
    inputs = data['inputs']
    targets = data['targets']
    positions = data['positions']
    metadata = data['metadata']
    
    print(f"  Inputs: {inputs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Positions: {positions.shape}")
    
    # Check that positions are sorted and in [0, 1]
    for i in range(min(5, positions.shape[0])):
        pos = positions[i]
        assert torch.all(pos[1:] >= pos[:-1]), f"Positions not sorted in sample {i}"
        assert pos.min() >= 0 and pos.max() <= 1, f"Positions out of [0,1] range in sample {i}"
    
    # Check that all data is finite
    assert torch.all(torch.isfinite(inputs)), "All inputs should be finite"
    assert torch.all(torch.isfinite(targets)), "All targets should be finite"
    assert torch.all(torch.isfinite(positions)), "All positions should be finite"
    
    # Check data ranges
    print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    print("  ✓ Irregular sampling data is valid")


def main():
    """Run all tests."""
    print("="*50)
    print("TESTING SYNTHETIC DATASETS")
    print("="*50)
    
    try:
        test_heat_equation_data()
        print()
        
        test_delayed_copy_data()
        print()
        
        test_irregular_sampling_data()
        print()
        
        print("="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 
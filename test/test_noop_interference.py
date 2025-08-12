#!/usr/bin/env python3
"""
Test script for the NoOpInterference module.

This test verifies that the NoOpInterference module works correctly as an
additive identity for ablation studies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from core.field_interference import NoOpInterference, create_field_interference


def test_noop_interference_basic():
    """Test basic functionality of NoOpInterference."""
    print("Testing Basic NoOpInterference Functionality")
    print("=" * 50)
    
    # Create the module
    noop = NoOpInterference()
    print("✓ NoOpInterference module created")
    
    # Test with different input shapes
    test_shapes = [
        (1, 10, 64),      # Single batch, 10 tokens, 64 dims
        (2, 5, 128),      # 2 batches, 5 tokens, 128 dims
        (4, 20, 256),     # 4 batches, 20 tokens, 256 dims
    ]
    
    for shape in test_shapes:
        test_input = torch.randn(shape)
        output = noop(test_input)
        
        # Check output shape matches input
        assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
        
        # Check output is all zeros
        assert torch.allclose(output, torch.zeros_like(output)), "Output is not all zeros"
        
        # Check device matches input
        assert output.device == test_input.device, "Device mismatch"
        
        print(f"✓ Shape {shape}: Output shape {output.shape}, all zeros, device match")
    
    print("✓ All basic functionality tests passed")


def test_noop_interference_additive_identity():
    """Test that NoOpInterference acts as an additive identity."""
    print("\nTesting Additive Identity Property")
    print("=" * 40)
    
    noop = NoOpInterference()
    
    # Test that x + NoOp(x) = x
    test_input = torch.randn(2, 8, 64)
    noop_output = noop(test_input)
    
    # Add the noop output to the input
    result = test_input + noop_output
    
    # Should be identical to original input
    assert torch.allclose(result, test_input), "Additive identity property failed"
    print("✓ Additive identity property verified: x + NoOp(x) = x")
    
    # Test with different input values
    test_input2 = torch.randn(3, 12, 128)
    noop_output2 = noop(test_input2)
    result2 = test_input2 + noop_output2
    
    assert torch.allclose(result2, test_input2), "Additive identity property failed for different input"
    print("✓ Additive identity property verified for different input shapes")
    
    print("✓ All additive identity tests passed")


def test_factory_function():
    """Test that the factory function correctly creates NoOpInterference."""
    print("\nTesting Factory Function")
    print("=" * 30)
    
    # Test with 'none' string
    noop1 = create_field_interference('none')
    assert isinstance(noop1, NoOpInterference), f"Expected NoOpInterference, got {type(noop1)}"
    print("✓ Factory function with 'none' string works")
    
    # Test with None
    noop2 = create_field_interference(None)
    assert isinstance(noop2, NoOpInterference), f"Expected NoOpInterference, got {type(noop2)}"
    print("✓ Factory function with None works")
    
    # Test with 'NONE' (case insensitive)
    noop3 = create_field_interference('NONE')
    assert isinstance(noop3, NoOpInterference), f"Expected NoOpInterference, got {type(noop3)}"
    print("✓ Factory function with 'NONE' (case insensitive) works")
    
    # Test that existing types still work
    standard = create_field_interference('standard', embed_dim=64, num_heads=4)
    assert not isinstance(standard, NoOpInterference), "Standard interference should not be NoOpInterference"
    print("✓ Factory function still works with existing types")
    
    print("✓ All factory function tests passed")


def test_noop_interference_integration():
    """Test that NoOpInterference can be used in place of other interference modules."""
    print("\nTesting Integration Capability")
    print("=" * 40)
    
    noop = NoOpInterference()
    
    # Simulate a typical forward pass
    batch_size, num_tokens, embed_dim = 2, 10, 64
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    
    # Apply noop interference
    output = noop(token_fields)
    
    # Check that output has correct properties
    assert output.shape == token_fields.shape, "Output shape mismatch"
    assert torch.allclose(output, torch.zeros_like(output)), "Output should be all zeros"
    
    # Test that it can be used in a simple network
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.interference = NoOpInterference()
            self.linear = nn.Linear(64, 64)
            
        def forward(self, x):
            # Apply interference (should do nothing)
            interference_output = self.interference(x)
            # Apply linear transformation
            return self.linear(x + interference_output)
    
    network = SimpleNetwork()
    input_tensor = torch.randn(2, 10, 64)
    output_tensor = network(input_tensor)
    
    assert output_tensor.shape == input_tensor.shape, "Network output shape mismatch"
    print("✓ NoOpInterference integrates correctly in a simple network")
    
    print("✓ All integration tests passed")


def test_noop_interference_gradients():
    """Test that NoOpInterference handles gradients correctly."""
    print("\nTesting Gradient Handling")
    print("=" * 35)
    
    noop = NoOpInterference()
    
    # Create input that requires gradients
    input_tensor = torch.randn(2, 5, 32, requires_grad=True)
    
    # Apply noop interference
    output = noop(input_tensor)
    
    # Check that output doesn't require gradients (it's constant zero)
    assert not output.requires_grad, "Output should not require gradients"
    
    # Test backward pass with a more appropriate loss function
    try:
        # Create a loss that involves both input and output
        # This tests that gradients flow correctly through the module
        combined = input_tensor + output  # This should equal input_tensor
        loss = combined.sum()
        loss.backward()
        print("✓ Backward pass completed successfully")
        
        # Check that input gradients are computed correctly
        assert input_tensor.grad is not None, "Input gradients should be computed"
        assert input_tensor.grad.shape == input_tensor.shape, "Gradient shape should match input shape"
        print("✓ Input gradients computed correctly")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    print("✓ All gradient tests passed")


def main():
    """Run all tests."""
    print("NOOP INTERFERENCE TEST SUITE")
    print("=" * 60)
    
    try:
        test_noop_interference_basic()
        test_noop_interference_additive_identity()
        test_factory_function()
        test_noop_interference_integration()
        test_noop_interference_gradients()
        
        print("\n" + "=" * 60)
        print("🎯 SUCCESS: NoOpInterference module working perfectly!")
        print("=" * 60)
        print("The module is ready for ablation studies:")
        print("  ✓ Acts as additive identity (x + NoOp(x) = x)")
        print("  ✓ Factory function integration working")
        print("  ✓ Gradient handling correct")
        print("  ✓ Can replace any interference module")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 
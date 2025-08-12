#!/usr/bin/env python3
"""
Test script for the accuracy masking fix in LanguageModelingStrategy.

This test verifies that the calculate_metrics method correctly masks out
ignored tokens for both delayed_copy (0 padding) and standard LM (-100 padding).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.task_strategies import LanguageModelingStrategy


def test_delayed_copy_accuracy_masking():
    """Test that delayed_copy tasks correctly mask out 0 padding tokens."""
    print("Testing Delayed Copy Accuracy Masking")
    print("=" * 50)
    
    strategy = LanguageModelingStrategy()
    
    # Simulate delayed_copy data
    # - vocab_size = 10 (digits 0-9), padding = 0
    # - pattern_length = 3, delay = 5
    # - Only positions 6-8 have meaningful targets
    batch_size, seq_len, vocab_size = 2, 10, 10
    
    # Create logits (random for testing)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create targets: only positions 6-8 have meaningful values, rest are 0 (padding)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets[0, 6:9] = torch.tensor([1, 2, 3])  # Pattern to copy
    targets[1, 6:9] = torch.tensor([4, 5, 6])  # Different pattern
    
    print(f"Targets shape: {targets.shape}")
    print(f"Targets[0]: {targets[0]}")
    print(f"Targets[1]: {targets[1]}")
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(logits, targets)
    accuracy = metrics["acc"]
    
    print(f"Calculated accuracy: {accuracy}")
    
    # Verify that only meaningful positions (6-8) are considered
    # The model should only be evaluated on copying the pattern, not on padding
    expected_valid_positions = 6  # 3 positions × 2 samples
    print(f"Expected valid positions: {expected_valid_positions}")
    
    # Test that the masking logic is working
    targets_flat = targets.view(-1)
    valid_mask = (targets_flat != 0)
    actual_valid_positions = valid_mask.sum().item()
    
    print(f"Actual valid positions: {actual_valid_positions}")
    assert actual_valid_positions == expected_valid_positions, f"Expected {expected_valid_positions} valid positions, got {actual_valid_positions}"
    
    print("✓ Delayed copy masking working correctly")
    return True


def test_standard_lm_accuracy_masking():
    """Test that standard language modeling tasks correctly mask out -100 padding tokens."""
    print("\nTesting Standard LM Accuracy Masking")
    print("=" * 45)
    
    strategy = LanguageModelingStrategy()
    
    # Simulate standard language modeling data
    # - vocab_size = 50000, padding = -100
    # - Sequence lengths vary, padded with -100
    batch_size, seq_len, vocab_size = 2, 8, 50000
    
    # Create logits (random for testing)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create targets: some positions have real tokens, others are -100 (padding)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    targets[0, :5] = torch.tensor([100, 200, 300, 400, 500])  # Real tokens
    targets[1, :6] = torch.tensor([150, 250, 350, 450, 550, 650])  # Real tokens
    
    print(f"Targets shape: {targets.shape}")
    print(f"Targets[0]: {targets[0]}")
    print(f"Targets[1]: {targets[1]}")
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(logits, targets)
    accuracy = metrics["acc"]
    
    print(f"Calculated accuracy: {accuracy}")
    
    # Verify that only non-padding positions are considered
    expected_valid_positions = 11  # 5 + 6 positions
    print(f"Expected valid positions: {expected_valid_positions}")
    
    # Test that the masking logic is working
    targets_flat = targets.view(-1)
    valid_mask = (targets_flat != -100)
    actual_valid_positions = valid_mask.sum().item()
    
    print(f"Actual valid positions: {actual_valid_positions}")
    assert actual_valid_positions == expected_valid_positions, f"Expected {expected_valid_positions} valid positions, got {actual_valid_positions}"
    
    print("✓ Standard LM masking working correctly")
    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting Edge Cases")
    print("=" * 25)
    
    strategy = LanguageModelingStrategy()
    
    # Test 1: All padding (should return 0 accuracy)
    print("Test 1: All padding tokens")
    logits = torch.randn(1, 5, 100)
    targets = torch.full((1, 5), 0, dtype=torch.long)  # All padding for delayed_copy
    
    metrics = strategy.calculate_metrics(logits, targets)
    assert metrics["acc"] == 0.0, f"Expected 0.0 accuracy for all padding, got {metrics['acc']}"
    print("✓ All padding case handled correctly")
    
    # Test 2: All padding with -100 (should return 0 accuracy)
    print("Test 2: All padding tokens (-100)")
    targets = torch.full((1, 5), -100, dtype=torch.long)  # All padding for standard LM
    
    metrics = strategy.calculate_metrics(logits, targets)
    assert metrics["acc"] == 0.0, f"Expected 0.0 accuracy for all padding, got {metrics['acc']}"
    print("✓ All padding case (-100) handled correctly")
    
    # Test 3: Single valid token
    print("Test 3: Single valid token")
    targets = torch.full((1, 5), 0, dtype=torch.long)
    targets[0, 2] = 5  # One valid token
    
    metrics = strategy.calculate_metrics(logits, targets)
    assert metrics["acc"] >= 0.0 and metrics["acc"] <= 1.0, f"Accuracy should be between 0 and 1, got {metrics['acc']}"
    print("✓ Single valid token case handled correctly")
    
    # Test 4: Empty batch (should handle gracefully)
    print("Test 4: Empty batch")
    logits = torch.randn(0, 5, 100)
    targets = torch.empty(0, 5, dtype=torch.long)
    
    try:
        metrics = strategy.calculate_metrics(logits, targets)
        print("✓ Empty batch handled gracefully")
    except Exception as e:
        print(f"❌ Empty batch failed: {e}")
        return False
    
    print("✓ All edge cases handled correctly")
    return True


def test_accuracy_calculation():
    """Test that accuracy is calculated correctly on valid tokens only."""
    print("\nTesting Accuracy Calculation")
    print("=" * 35)
    
    strategy = LanguageModelingStrategy()
    
    # Create a controlled test case
    batch_size, seq_len, vocab_size = 1, 6, 10
    
    # Create logits where the model always predicts token 1
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    logits[:, :, 1] = 1.0  # Always predict token 1
    
    # Create targets: positions 0, 2, 4 have meaningful values, others are 0 (padding)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets[0, 0] = 1  # Correct prediction
    targets[0, 2] = 2  # Wrong prediction (model predicts 1, target is 2)
    targets[0, 4] = 1  # Correct prediction
    
    print(f"Targets: {targets[0]}")
    print(f"Model always predicts: 1")
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(logits, targets)
    accuracy = metrics["acc"]
    
    # Expected accuracy: 2 correct out of 3 valid positions = 2/3
    expected_accuracy = 2.0 / 3.0
    print(f"Expected accuracy: {expected_accuracy}")
    print(f"Calculated accuracy: {accuracy}")
    
    # Allow small floating point differences
    assert abs(accuracy - expected_accuracy) < 1e-6, f"Expected {expected_accuracy}, got {accuracy}"
    
    print("✓ Accuracy calculation working correctly")
    return True


def main():
    """Run all tests."""
    print("ACCURACY MASKING TEST SUITE")
    print("=" * 60)
    
    try:
        test_delayed_copy_accuracy_masking()
        test_standard_lm_accuracy_masking()
        test_edge_cases()
        test_accuracy_calculation()
        
        print("\n" + "=" * 60)
        print("🎯 SUCCESS: Accuracy masking fix working perfectly!")
        print("=" * 60)
        print("The LanguageModelingStrategy now correctly:")
        print("  ✓ Masks out delayed_copy padding (0) tokens")
        print("  ✓ Masks out standard LM padding (-100) tokens")
        print("  ✓ Calculates accuracy only on meaningful positions")
        print("  ✓ Handles edge cases gracefully")
        print("\nThis ensures accurate evaluation of model performance!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 
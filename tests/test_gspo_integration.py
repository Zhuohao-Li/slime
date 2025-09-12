#!/usr/bin/env python3
"""
Integration tests for GSPO implementation across backends.

This module provides end-to-end testing of GSPO functionality and validates
consistency between different backend implementations.
"""

import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add slime to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_mock_batch(batch_size=2, seq_len=6):
    """Create mock batch data for testing."""
    # Create realistic log probabilities
    log_probs = torch.randn(batch_size, seq_len, dtype=torch.float32) * 0.5 - 1.5
    old_log_probs = log_probs + torch.randn_like(log_probs) * 0.1  # Small perturbation
    
    # Create loss masks with some padding
    loss_masks = torch.ones(batch_size, seq_len, dtype=torch.float32)
    # Add some padding to second sample
    if batch_size > 1 and seq_len > 3:
        loss_masks[1, -2:] = 0.0  # Mask last 2 tokens
    
    # Create advantages
    advantages = torch.randn(batch_size, seq_len, dtype=torch.float32) * 0.5
    
    # Create tokens (not used in KL computation but needed for batch structure)
    tokens = torch.randint(0, 1000, (batch_size, seq_len + 1))  # +1 for input_ids structure
    
    batch = {
        "log_probs": log_probs,
        "loss_masks": loss_masks,
        "advantages": advantages,
        "tokens": tokens,
        "ref_log_probs": old_log_probs + torch.randn_like(old_log_probs) * 0.05  # For KL loss testing
    }
    
    return batch, log_probs, old_log_probs, loss_masks


def test_fsdp_gspo_implementation():
    """Test FSDP GSPO implementation with realistic data."""
    try:
        from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
    except ImportError:
        print("FSDP backend not available, skipping FSDP tests")
        return False
    
    print("Testing FSDP GSPO implementation...")
    
    # Create mock actor
    actor = FSDPTrainRayActor(
        world_size=1, 
        rank=0, 
        master_addr="localhost", 
        master_port=29500, 
        wandb_run_id="test"
    )
    
    # Create test data
    batch, current_log_probs, old_log_probs, loss_masks = create_mock_batch()
    
    # Test GSPO KL computation
    gspo_kl = actor._compute_gspo_kl_divergence(
        current_log_probs, old_log_probs, loss_masks
    )
    
    # Validate results
    assert gspo_kl.shape == current_log_probs.shape, "GSPO KL shape mismatch"
    assert torch.isfinite(gspo_kl).all(), "GSPO KL contains non-finite values"
    
    # Check that each sequence has uniform KL values
    for i in range(gspo_kl.shape[0]):
        valid_mask = loss_masks[i] > 0
        if valid_mask.sum() > 0:
            # All valid tokens in a sequence should have the same KL value
            valid_kl_values = gspo_kl[i][valid_mask]
            assert torch.allclose(valid_kl_values, valid_kl_values[0], atol=1e-6), \
                f"Sample {i}: GSPO KL values not uniform across tokens"
    
    print("✓ FSDP GSPO implementation test passed")
    return True


def test_gspo_vs_grpo_comparison():
    """Compare GSPO vs GRPO behavior."""
    try:
        from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
    except ImportError:
        print("FSDP backend not available, skipping comparison tests")
        return False
    
    print("Testing GSPO vs GRPO comparison...")
    
    # Create mock actor
    actor = FSDPTrainRayActor(
        world_size=1, 
        rank=0, 
        master_addr="localhost", 
        master_port=29500, 
        wandb_run_id="test"
    )
    
    # Create test data with high variance in token-level KL
    batch_size, seq_len = 2, 5
    current_log_probs = torch.tensor([
        [-1.0, -3.0, -1.5, -2.5, -1.2],  # Sample 1
        [-2.0, -1.0, -2.2, -1.1, -2.1]   # Sample 2
    ], dtype=torch.float32)
    
    old_log_probs = torch.tensor([
        [-1.5, -1.0, -2.0, -1.5, -1.8],  # Sample 1
        [-1.8, -2.0, -1.9, -2.1, -1.9]   # Sample 2
    ], dtype=torch.float32)
    
    loss_masks = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],  # All valid
        [1.0, 1.0, 1.0, 0.0, 0.0]   # Last 2 masked
    ], dtype=torch.float32)
    
    # Compute GSPO KL
    gspo_kl = actor._compute_gspo_kl_divergence(current_log_probs, old_log_probs, loss_masks)
    
    # Compute GRPO KL (token-level)
    grpo_kl = old_log_probs - current_log_probs
    
    # Verify differences
    assert not torch.allclose(gspo_kl, grpo_kl, atol=1e-4), \
        "GSPO and GRPO should produce different results with varying token KL"
    
    # Check GSPO uniformity within sequences
    for i in range(batch_size):
        valid_mask = loss_masks[i] > 0
        if valid_mask.sum() > 1:
            gspo_values = gspo_kl[i][valid_mask]
            assert torch.allclose(gspo_values, gspo_values[0], atol=1e-6), \
                f"GSPO values not uniform for sample {i}"
    
    # Check that GRPO has variance within sequences
    for i in range(batch_size):
        valid_mask = loss_masks[i] > 0
        if valid_mask.sum() > 1:
            grpo_values = grpo_kl[i][valid_mask]
            grpo_variance = torch.var(grpo_values)
            assert grpo_variance > 1e-6, f"GRPO should have variance within sample {i}"
    
    print("✓ GSPO vs GRPO comparison test passed")
    return True


def test_megatron_gspo_consistency():
    """Test consistency with Megatron GSPO implementation logic."""
    try:
        from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
    except ImportError:
        print("FSDP backend not available, skipping consistency tests")
        return False
    
    print("Testing consistency with Megatron GSPO logic...")
    
    # Create mock actor
    actor = FSDPTrainRayActor(
        world_size=1, 
        rank=0, 
        master_addr="localhost", 
        master_port=29500, 
        wandb_run_id="test"
    )
    
    # Test data
    batch_size, seq_len = 1, 4
    current_log_probs = torch.tensor([[-1.0, -1.5, -2.0, -1.2]], dtype=torch.float32)
    old_log_probs = torch.tensor([[-1.1, -1.4, -2.1, -1.3]], dtype=torch.float32)
    loss_masks = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    
    # Compute using FSDP implementation
    fsdp_result = actor._compute_gspo_kl_divergence(current_log_probs, old_log_probs, loss_masks)
    
    # Manually compute using Megatron-style logic
    token_kl = old_log_probs - current_log_probs
    sequence_avg_kl = (token_kl * loss_masks).sum() / torch.clamp_min(loss_masks.sum(), 1)
    expected_result = sequence_avg_kl.expand_as(current_log_probs)
    
    # Should match exactly
    torch.testing.assert_close(fsdp_result, expected_result, atol=1e-7, rtol=1e-7)
    
    print("✓ Megatron GSPO consistency test passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("Running GSPO integration tests...")
    print("=" * 50)
    
    test_results = []
    
    # Test FSDP implementation
    test_results.append(test_fsdp_gspo_implementation())
    
    # Test GSPO vs GRPO comparison
    test_results.append(test_gspo_vs_grpo_comparison())
    
    # Test Megatron consistency
    test_results.append(test_megatron_gspo_consistency())
    
    print("=" * 50)
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Integration tests summary: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

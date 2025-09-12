#!/usr/bin/env python3
"""
Unit tests for FSDP backend GSPO implementation.

This module tests the GSPO (Group Sequence Policy Optimization) functionality
in the FSDP backend to ensure correctness and consistency.
"""

import torch
import pytest
from unittest.mock import MagicMock
import sys
import os

# Add slime to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


@pytest.mark.skipif(not FSDP_AVAILABLE, reason="FSDP backend not available")
class TestFSDPGSPO:
    """Test suite for FSDP GSPO implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock args
        self.args = MagicMock()
        self.args.advantage_estimator = "gspo"
        
        # Create mock actor instance
        self.actor = FSDPTrainRayActor(
            world_size=1, 
            rank=0, 
            master_addr="localhost", 
            master_port=29500, 
            wandb_run_id="test"
        )
    
    def test_gspo_kl_computation_single_sample(self):
        """Test GSPO KL computation for a single sample."""
        # Create test data
        batch_size = 1
        seq_len = 5
        
        # Mock log probabilities - current model slightly different from old model
        log_probs = torch.tensor([[-1.0, -1.5, -2.0, -1.2, -1.8]], dtype=torch.float32)
        old_log_probs = torch.tensor([[-1.1, -1.4, -2.1, -1.3, -1.7]], dtype=torch.float32)
        
        # Loss mask - all tokens are valid
        loss_masks = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # Compute GSPO KL
        result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        # Verify shape
        assert result.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {result.shape}"
        
        # Manually compute expected result
        token_kl = old_log_probs - log_probs  # [-0.1, 0.1, -0.1, -0.1, 0.1]
        expected_avg_kl = (token_kl * loss_masks).sum() / loss_masks.sum()  # -0.02
        
        # All tokens should have the same sequence-level KL
        expected_result = torch.full_like(log_probs, expected_avg_kl)
        
        torch.testing.assert_close(result, expected_result, atol=1e-6, rtol=1e-6)
    
    def test_gspo_kl_computation_with_padding(self):
        """Test GSPO KL computation with padded sequences."""
        batch_size = 2
        seq_len = 4
        
        # Create test data with different valid lengths
        log_probs = torch.tensor([
            [-1.0, -1.5, -2.0, -1.2],  # Sample 1: all valid
            [-1.1, -1.4, -2.1, -1.3]   # Sample 2: first 2 valid
        ], dtype=torch.float32)
        
        old_log_probs = torch.tensor([
            [-1.1, -1.4, -2.1, -1.3],  # Sample 1
            [-1.0, -1.5, -2.0, -1.2]   # Sample 2
        ], dtype=torch.float32)
        
        # Loss masks - sample 1 all valid, sample 2 only first 2 tokens valid
        loss_masks = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],  # All valid
            [1.0, 1.0, 0.0, 0.0]   # Only first 2 valid
        ], dtype=torch.float32)
        
        # Compute GSPO KL
        result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        # Verify shape
        assert result.shape == (batch_size, seq_len)
        
        # Manually compute expected results for each sample
        # Sample 1: all tokens valid
        sample1_kl = old_log_probs[0] - log_probs[0]  # [-0.1, 0.1, -0.1, -0.1]
        sample1_avg_kl = (sample1_kl * loss_masks[0]).sum() / loss_masks[0].sum()  # -0.05
        
        # Sample 2: only first 2 tokens valid
        sample2_kl = old_log_probs[1] - log_probs[1]  # [0.1, -0.1, 0.1, 0.1]
        sample2_avg_kl = (sample2_kl * loss_masks[1]).sum() / loss_masks[1].sum()  # 0.0
        
        # Check each sample separately
        torch.testing.assert_close(result[0], torch.full_like(log_probs[0], sample1_avg_kl), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result[1], torch.full_like(log_probs[1], sample2_avg_kl), atol=1e-6, rtol=1e-6)
    
    def test_gspo_vs_grpo_difference(self):
        """Test that GSPO produces different results than GRPO for varying token-level KL."""
        batch_size = 1
        seq_len = 4
        
        # Create data where token-level KL varies significantly
        log_probs = torch.tensor([[-1.0, -3.0, -1.0, -3.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[-2.0, -1.0, -2.0, -1.0]], dtype=torch.float32)
        loss_masks = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # GSPO result
        gspo_result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        # GRPO result (token-level KL)
        grpo_result = old_log_probs - log_probs
        
        # Results should be different
        assert not torch.allclose(gspo_result, grpo_result), "GSPO and GRPO should produce different results"
        
        # GSPO should have uniform values across tokens for each sequence
        assert torch.allclose(gspo_result[0, 0], gspo_result[0, :]), "GSPO should have uniform KL across tokens"
        
        # GRPO should have varying values
        assert not torch.allclose(grpo_result[0, 0], grpo_result[0, :]), "GRPO should have varying KL across tokens"
    
    def test_gspo_edge_case_empty_sequence(self):
        """Test GSPO with empty sequences (all tokens masked out)."""
        batch_size = 1
        seq_len = 3
        
        log_probs = torch.tensor([[-1.0, -1.5, -2.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[-1.1, -1.4, -2.1]], dtype=torch.float32)
        
        # All tokens masked out
        loss_masks = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        
        # Should not crash and should return zeros
        result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        # With torch.clamp_min(mask.sum(), 1), we expect 0/1 = 0
        expected = torch.zeros_like(log_probs)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    
    def test_gspo_single_token_sequence(self):
        """Test GSPO with single token sequences."""
        batch_size = 2
        seq_len = 1
        
        log_probs = torch.tensor([[-1.0], [-2.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[-1.5], [-1.8]], dtype=torch.float32)
        loss_masks = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        
        result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        # For single token, GSPO should equal GRPO
        expected = old_log_probs - log_probs
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    
    def test_gspo_numerical_stability(self):
        """Test GSPO numerical stability with extreme values."""
        batch_size = 1
        seq_len = 3
        
        # Use extreme values to test numerical stability
        log_probs = torch.tensor([[-100.0, -0.001, -50.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[-99.0, -0.002, -49.0]], dtype=torch.float32)
        loss_masks = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # Should not produce NaN or inf
        result = self.actor._compute_gspo_kl_divergence(log_probs, old_log_probs, loss_masks)
        
        assert torch.isfinite(result).all(), "GSPO result should be finite"
        assert not torch.isnan(result).any(), "GSPO result should not contain NaN"


def test_gspo_import():
    """Test that GSPO-related imports work correctly."""
    if FSDP_AVAILABLE:
        from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
        
        # Check that the method exists
        assert hasattr(FSDPTrainRayActor, '_compute_gspo_kl_divergence'), \
            "FSDPTrainRayActor should have _compute_gspo_kl_divergence method"


if __name__ == "__main__":
    # Run tests
    if FSDP_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("FSDP backend not available, skipping tests")

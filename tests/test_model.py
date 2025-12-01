"""
Tests for the PPO model architecture and initialization.

These tests verify that the model correctly initializes weights
to produce non-saturated outputs for fold probabilities.
"""

import pytest
import torch
import numpy as np

from training.ppo_model import PokerPPOModel
from simulation.poker_env import (
    CARD_ENCODING_DIM,
    NUM_CARD_SLOTS,
    NUM_HAND_FEATURES,
    MAX_PLAYERS,
    FEATURES_PER_PLAYER,
    GLOBAL_NUMERIC_FEATURES,
)


@pytest.fixture
def model():
    """Create a fresh model instance for testing."""
    return PokerPPOModel()


@pytest.fixture
def obs_dim():
    """Calculate the observation dimension."""
    return (
        NUM_CARD_SLOTS * CARD_ENCODING_DIM +
        NUM_HAND_FEATURES +
        MAX_PLAYERS * FEATURES_PER_PLAYER +
        GLOBAL_NUMERIC_FEATURES
    )


def test_model_initialization_produces_non_saturated_fold_probs(model, obs_dim):
    """
    GIVEN a freshly initialized PokerPPOModel
    WHEN random observations are passed through
    THEN fold probabilities should NOT be saturated (≈0 or ≈1)
    
    This tests the fix for binary fold decisions.
    Initial fold probabilities should be near 0.5, not extreme values.
    """
    model.eval()
    
    # Generate random observations
    batch_size = 100
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    with torch.no_grad():
        fold_logit, _, _, _ = model.forward(dummy_obs)
        fold_prob = torch.sigmoid(fold_logit)
    
    # Check that fold probabilities are not saturated
    # With proper initialization, most should be between 0.2 and 0.8
    fold_probs_np = fold_prob.numpy().flatten()
    
    # Count how many are in the "non-saturated" range
    non_saturated = np.sum((fold_probs_np > 0.1) & (fold_probs_np < 0.9))
    saturation_rate = 1 - (non_saturated / batch_size)
    
    # With small initialization gains, most outputs should be near 0.5
    # Allow up to 20% to be outside the range (statistical variance)
    assert saturation_rate < 0.2, \
        f"Too many saturated fold probabilities: {saturation_rate*100:.1f}% " \
        f"(expected < 20%). Min: {fold_probs_np.min():.3f}, Max: {fold_probs_np.max():.3f}"


def test_model_fold_head_weights_are_small(model):
    """
    GIVEN a freshly initialized PokerPPOModel
    WHEN checking the fold head final layer weights
    THEN they should be small (gain=0.01)
    
    This verifies the weight initialization fix.
    """
    # Get the final layer of fold head
    fold_final_layer = model.fold_head[-1]
    
    # Check that weights are small
    weight_std = fold_final_layer.weight.std().item()
    
    # With orthogonal init and gain=0.01, std should be very small
    assert weight_std < 0.1, \
        f"Fold head weights too large (std={weight_std:.4f}), expected < 0.1"


def test_model_bet_head_weights_are_small(model):
    """
    GIVEN a freshly initialized PokerPPOModel
    WHEN checking the bet head final layer weights
    THEN they should be small (gain=0.01)
    """
    # Get the final layer of bet head
    bet_final_layer = model.bet_head[-1]
    
    # Check that weights are small
    weight_std = bet_final_layer.weight.std().item()
    
    assert weight_std < 0.1, \
        f"Bet head weights too large (std={weight_std:.4f}), expected < 0.1"


def test_model_forward_returns_expected_shapes(model, obs_dim):
    """
    GIVEN a PokerPPOModel
    WHEN forward is called with a batch of observations
    THEN output shapes should be correct
    """
    batch_size = 8
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    fold_logit, bet_alpha, bet_beta, value = model.forward(dummy_obs)
    
    assert fold_logit.shape == (batch_size, 1)
    assert bet_alpha.shape == (batch_size, 1)
    assert bet_beta.shape == (batch_size, 1)
    assert value.shape == (batch_size, 1)


def test_model_get_action_and_value_returns_valid_actions(model, obs_dim):
    """
    GIVEN a PokerPPOModel
    WHEN get_action_and_value is called
    THEN actions should be in [0, 1] range
    """
    batch_size = 8
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs)
    
    assert action.shape == (batch_size, 2)
    
    # Check actions are in valid range [0, 1]
    assert (action[:, 0] >= 0).all() and (action[:, 0] <= 1).all(), \
        "p_fold action should be in [0, 1]"
    assert (action[:, 1] >= 0).all() and (action[:, 1] <= 1).all(), \
        "bet_scalar action should be in [0, 1]"


def test_model_beta_distribution_params_are_positive(model, obs_dim):
    """
    GIVEN a PokerPPOModel
    WHEN forward is called
    THEN Beta distribution parameters (alpha, beta) should be > 0
    """
    batch_size = 8
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    _, bet_alpha, bet_beta, _ = model.forward(dummy_obs)
    
    # Beta params should be positive (we add 1.0 after softplus)
    assert (bet_alpha > 0).all(), "bet_alpha should be positive"
    assert (bet_beta > 0).all(), "bet_beta should be positive"
    
    # Actually should be >= 1.0 due to the +1.0 in the model
    assert (bet_alpha >= 1.0).all(), "bet_alpha should be >= 1.0"
    assert (bet_beta >= 1.0).all(), "bet_beta should be >= 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

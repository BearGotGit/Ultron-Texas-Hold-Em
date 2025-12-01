"""
Test script to verify the updated PPO model architecture.

This script:
1. Creates a PokerEnv instance
2. Instantiates the PPO model
3. Generates a sample observation
4. Performs a forward pass
5. Verifies output shapes and parameter count

Run from repo root:
    python -m tests.manual.test_architecture
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from simulation.poker_env import PokerEnv, PokerEnvConfig
from training.ppo_model import PokerPPOModel
from agents.monte_carlo_agent import MonteCarloAgent, RandomAgent


def test_model_architecture():
    """Test that the model architecture works correctly."""
    
    print("="*60)
    print("Testing Updated PPO Model Architecture")
    print("="*60)
    
    # Create environment
    print("\n1. Creating PokerEnv...")
    config = PokerEnvConfig()
    players = [
        RandomAgent(f"Player_{i}", config.starting_stack)
        for i in range(6)
    ]
    env = PokerEnv(players=players, config=config, hero_idx=0, render_mode=None)
    obs, info = env.reset()
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Expected: (423,)")
    
    # Create model
    print("\n2. Creating PPO model...")
    model = PokerPPOModel(
        card_embed_dim=64,
        hidden_dim=256,
        num_shared_layers=2,
    )
    print(f"   ✓ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # Add batch dimension
    print(f"   Input tensor shape: {obs_tensor.shape}")
    
    with torch.no_grad():
        fold_logit, bet_alpha, bet_beta, value = model(obs_tensor)
    
    print(f"   ✓ fold_logit shape: {fold_logit.shape}")
    print(f"   ✓ bet_alpha shape: {bet_alpha.shape}")
    print(f"   ✓ bet_beta shape: {bet_beta.shape}")
    print(f"   ✓ value shape: {value.shape}")
    
    # Verify output shapes
    print("\n4. Verifying output shapes...")
    assert fold_logit.shape == (1, 1), f"Expected (1, 1), got {fold_logit.shape}"
    assert bet_alpha.shape == (1, 1), f"Expected (1, 1), got {bet_alpha.shape}"
    assert bet_beta.shape == (1, 1), f"Expected (1, 1), got {bet_beta.shape}"
    assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"
    print("   ✓ All output shapes correct!")
    
    # Verify output values are reasonable
    print("\n5. Verifying output values...")
    print(f"   fold_logit: {fold_logit.item():.4f}")
    print(f"   bet_alpha: {bet_alpha.item():.4f} (should be >= 1.0)")
    print(f"   bet_beta: {bet_beta.item():.4f} (should be >= 1.0)")
    print(f"   value: {value.item():.4f}")
    
    assert bet_alpha.item() >= 1.0, f"bet_alpha should be >= 1.0, got {bet_alpha.item()}"
    assert bet_beta.item() >= 1.0, f"bet_beta should be >= 1.0, got {bet_beta.item()}"
    print("   ✓ All output values within expected ranges!")
    
    # Test action sampling
    print("\n6. Testing action sampling...")
    p_fold = torch.sigmoid(fold_logit)
    fold_dist = torch.distributions.Bernoulli(probs=p_fold)
    fold_action = fold_dist.sample()
    
    bet_dist = torch.distributions.Beta(bet_alpha, bet_beta)
    bet_scalar = bet_dist.sample()
    
    print(f"   p_fold: {p_fold.item():.4f}")
    print(f"   fold_action: {fold_action.item()}")
    print(f"   bet_scalar: {bet_scalar.item():.4f}")
    print("   ✓ Action sampling successful!")
    
    # Test batch processing
    print("\n7. Testing batch processing...")
    batch_size = 32
    obs_batch = torch.from_numpy(np.stack([obs] * batch_size))
    print(f"   Batch input shape: {obs_batch.shape}")
    
    with torch.no_grad():
        fold_logit_batch, bet_alpha_batch, bet_beta_batch, value_batch = model(obs_batch)
    
    print(f"   ✓ fold_logit_batch shape: {fold_logit_batch.shape}")
    print(f"   ✓ bet_alpha_batch shape: {bet_alpha_batch.shape}")
    print(f"   ✓ bet_beta_batch shape: {bet_beta_batch.shape}")
    print(f"   ✓ value_batch shape: {value_batch.shape}")
    
    assert fold_logit_batch.shape == (batch_size, 1)
    assert bet_alpha_batch.shape == (batch_size, 1)
    assert bet_beta_batch.shape == (batch_size, 1)
    assert value_batch.shape == (batch_size, 1)
    print("   ✓ Batch processing works correctly!")
    
    # Print architecture summary
    print("\n" + "="*60)
    print("Architecture Summary")
    print("="*60)
    print(f"Card Embedding: 53 → 64 → 64 (shared across 7 cards)")
    print(f"Hand Embedding: 10 → 32 → 32")
    print(f"Numeric Embedding: 42 → 64 → 64")
    print(f"Combined Features: 448 + 32 + 64 = 544")
    print(f"Shared Trunk: 544 → 256 → 256")
    print(f"Fold Head: 256 → 128 → 1 (sigmoid)")
    print(f"Bet Head: 256 → 128 → 2 (Beta params)")
    print(f"Value Head: 256 → 128 → 1")
    print(f"\nTotal Parameters: {total_params:,}")
    print("="*60)
    
    print("\n✅ All tests passed! Architecture is working correctly.")
    print()


if __name__ == "__main__":
    test_model_architecture()

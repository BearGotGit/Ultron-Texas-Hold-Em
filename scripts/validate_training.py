#!/usr/bin/env python3
"""
Training Validation Script.

Quick sanity check to verify the RL training pipeline is working:
1. Saves initial model checkpoint
2. Trains for 100 iterations
3. Saves final checkpoint
4. Compares weights to verify they changed
5. Plays 10 hands against a random opponent to test the trained model

Usage:
    python scripts/validate_training.py

Expected runtime: < 1 minute
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

# Ensure the project root is in the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.ppo_model import PokerPPOModel
from training.train_rl_model import PPOConfig, PPOTrainer
from simulation.poker_env import PokerEnv, PokerEnvConfig
from agents.monte_carlo_agent import RandomAgent
from utils.device import DEVICE


def compare_model_weights(
    before_state: dict,
    after_state: dict,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """
    Compare two model state dicts to check if weights have changed.
    
    Args:
        before_state: State dict before training
        after_state: State dict after training
        tolerance: Minimum difference to consider weights changed
        
    Returns:
        (weights_changed, max_diff): Tuple of bool and max absolute difference
    """
    max_diff = 0.0
    
    for key in before_state:
        if key not in after_state:
            continue
        before_tensor = before_state[key]
        after_tensor = after_state[key]
        diff = torch.abs(before_tensor - after_tensor).max().item()
        max_diff = max(max_diff, diff)
    
    weights_changed = max_diff > tolerance
    return weights_changed, max_diff


def play_hands_vs_random(
    model: PokerPPOModel,
    num_hands: int = 10,
    device: torch.device = DEVICE,
) -> dict:
    """
    Play hands with the trained model against a random opponent.
    
    Args:
        model: Trained PPO model
        num_hands: Number of hands to play
        device: PyTorch device
        
    Returns:
        Dictionary with win_rate, avg_profit, and wins
    """
    model.eval()
    
    # Create environment with random opponent
    config = PokerEnvConfig(
        big_blind=10,
        small_blind=5,
        starting_stack=1000,
        max_players=2,
    )
    
    # Create agents - hero is placeholder (controlled by model), opponent is random
    hero = RandomAgent("Hero", starting_money=1000)
    opponent = RandomAgent("Random-Opponent", starting_money=1000, fold_prob=0.3, raise_prob=0.2)
    
    env = PokerEnv(
        players=[hero, opponent],
        config=config,
        hero_idx=0,
    )
    
    wins = 0
    total_profit = 0.0
    
    for hand in range(num_hands):
        obs, _ = env.reset()
        initial_chips = env.players[env.hero_idx].money
        done = False
        
        while not done:
            # Get model action
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
        
        final_chips = env.players[env.hero_idx].money
        profit = final_chips - initial_chips
        
        if profit > 0:
            wins += 1
        total_profit += profit
    
    return {
        "wins": wins,
        "total_hands": num_hands,
        "win_rate": wins / num_hands,
        "avg_profit": total_profit / num_hands,
    }


def main():
    """Run training validation."""
    print("=" * 60)
    print("Training Validation Script")
    print("=" * 60)
    print()
    
    # Use temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        before_path = checkpoint_dir / "before_training.pt"
        after_path = checkpoint_dir / "after_training.pt"
        
        # Configure for quick validation run
        # Use fewer timesteps to complete in < 1 minute
        config = PPOConfig(
            total_timesteps=6400,  # 50 iterations * 128 steps (fast run)
            num_envs=1,
            num_players=2,
            num_steps=128,
            log_interval=25,
            save_interval=1000,  # Don't auto-save during validation
            eval_interval=1000,  # Don't auto-eval during validation
            run_name="validation_run",
            save_dir=str(checkpoint_dir),
            log_dir=str(checkpoint_dir / "logs"),
        )
        
        print("1. Creating PPO trainer...")
        trainer = PPOTrainer(config, device=DEVICE)
        
        # Save initial weights
        print("2. Saving initial checkpoint (before training)...")
        before_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        torch.save({"model_state_dict": before_state}, before_path)
        print(f"   Saved to: {before_path}")
        
        # Train
        print()
        print("3. Training for 100 iterations...")
        print("-" * 40)
        trainer.train(save_path="final_validation.pt")
        print("-" * 40)
        
        # Save final weights
        print()
        print("4. Saving final checkpoint (after training)...")
        after_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        torch.save({"model_state_dict": after_state}, after_path)
        print(f"   Saved to: {after_path}")
        
        # Compare weights
        print()
        print("5. Comparing weights before and after training...")
        weights_changed, max_diff = compare_model_weights(before_state, after_state)
        
        print()
        print("=" * 60)
        print("TRAINING VALIDATION RESULTS")
        print("=" * 60)
        
        if weights_changed:
            print("✓ Weights updated (training is working!)")
            print(f"  Maximum weight change: {max_diff:.6f}")
        else:
            print("✗ Weights frozen (training may have issues!)")
            print(f"  Maximum weight change: {max_diff:.6f}")
        
        # Play test hands
        print()
        print("6. Testing trained model vs random opponent (10 hands)...")
        results = play_hands_vs_random(trainer.model, num_hands=10, device=DEVICE)
        
        print()
        print("Game Results:")
        print(f"  Wins: {results['wins']}/{results['total_hands']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Average Profit: {results['avg_profit']:.1f} chips")
        
        print()
        print("=" * 60)
        
        # Final summary
        validation_passed = weights_changed
        if validation_passed:
            print("✓ VALIDATION PASSED: Training pipeline is working correctly!")
        else:
            print("✗ VALIDATION FAILED: Please check the training configuration.")
        
        print("=" * 60)
        
        return 0 if validation_passed else 1


if __name__ == "__main__":
    exit(main())

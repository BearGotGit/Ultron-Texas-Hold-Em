"""
Behavior Cloning script for PokerPPOModel.

Loads saved rollouts (torch .pt files saved by the trainer) and trains
the actor heads of `PokerPPOModel` with supervised losses:
 - fold: BCEWithLogitsLoss on the fold logit
 - bet: MSE on the predicted bet mean vs target bet scalar

This produces a warm-start checkpoint you can resume PPO from.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from training.ppo_model import PokerPPOModel
from utils.device import DEVICE


def load_rollouts(rollout_dir: Path):
    files = sorted(rollout_dir.glob("*.pt"))
    obs_list = []
    actions_list = []
    for f in files:
        # Some rollouts may have been saved with custom classes pickled (PPOConfig)
        # Provide a placeholder in __main__ so torch can unpickle safely for inspection
        try:
            import __main__ as _m
            if not hasattr(_m, 'PPOConfig'):
                class PPOConfig:
                    pass
                setattr(_m, 'PPOConfig', PPOConfig)
        except Exception:
            pass
        data = torch.load(f, map_location="cpu", weights_only=False)
        obs = data.get("observations")  # (num_steps, num_envs, obs_dim)
        acts = data.get("actions")
        if obs is None or acts is None:
            continue
        # reshape to (num_steps * num_envs, obs_dim)
        s, e, d = obs.shape
        obs_list.append(obs.reshape(s * e, d))
        actions_list.append(acts.reshape(s * e, -1))

    if not obs_list:
        raise RuntimeError(f"No rollouts found in {rollout_dir}")

    observations = torch.cat(obs_list, dim=0).float()
    actions = torch.cat(actions_list, dim=0).float()
    return observations, actions


def train_bc(
    rollout_dir: str,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: torch.device = DEVICE,
    save_path: str = "checkpoints/bc_initial.pt",
):
    rollout_path = Path(rollout_dir)
    observations, actions = load_rollouts(rollout_path)
    dataset = TensorDataset(observations, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = observations.shape[1]

    model = PokerPPOModel().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for obs_b, acts_b in loader:
            obs_b = obs_b.to(device)
            acts_b = acts_b.to(device)

            fold_target = acts_b[:, 0:1]
            bet_target = acts_b[:, 1:2]

            fold_logit, bet_alpha, bet_beta, _ = model.forward(obs_b)

            # BCE on fold (use logits)
            fold_loss = bce(fold_logit, fold_target)

            # Use Beta mean as bet prediction: mean = alpha / (alpha + beta)
            bet_mean = bet_alpha / (bet_alpha + bet_beta)
            bet_loss = mse(bet_mean, bet_target)

            loss = fold_loss + bet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * obs_b.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs}  Avg loss: {avg_loss:.6f}")

    # Save warm-start model
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
    }, save_p)
    print(f"Saved behavior-cloned model to {save_p}")


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning from saved rollouts")
    parser.add_argument("--rollout-dir", type=str, default="rollouts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="checkpoints/bc_initial.pt")
    args = parser.parse_args()

    train_bc(
        rollout_dir=args.rollout_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

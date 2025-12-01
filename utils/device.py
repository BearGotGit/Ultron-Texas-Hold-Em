"""
Device configuration for PyTorch.
Automatically selects the best available device: MPS (Apple Silicon) > CUDA > CPU.

Note: MPS currently doesn't support Beta distribution sampling (used in PPO).
      Falling back to CPU until PyTorch adds support.
      See: https://github.com/pytorch/pytorch/issues/141287
"""

import os
import torch

# Enable MPS fallback to CPU for unsupported operations
# Must be set before any torch operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    
    Priority:
        1. CUDA (NVIDIA GPU)
        2. CPU (fallback) - MPS disabled due to Beta distribution issues
    
    Returns:
        torch.device: The selected device
    """
    # MPS doesn't support Beta distribution sampling yet
    # Uncomment when PyTorch adds support:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Global device constant
DEVICE = get_device()


if __name__ == "__main__":
    print(f"Selected device: {DEVICE}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

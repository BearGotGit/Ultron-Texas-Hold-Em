"""Inspect a PyTorch checkpoint safely by providing placeholder classes.

Usage:
    python scripts/inspect_checkpoint.py checkpoints/final.pt
"""
import sys
from pathlib import Path
import __main__ as _m

# Provide placeholder PPOConfig for older checkpoints
if not hasattr(_m, 'PPOConfig'):
    class PPOConfig:
        pass
    setattr(_m, 'PPOConfig', PPOConfig)

import torch

if len(sys.argv) < 2:
    print('Usage: python scripts/inspect_checkpoint.py <path>')
    sys.exit(1)

path = Path(sys.argv[1])
if not path.exists():
    print('Checkpoint not found:', path)
    sys.exit(2)

ck = torch.load(str(path), map_location='cpu', weights_only=False)
print('Keys:', list(ck.keys()))
print('global_step:', ck.get('global_step'))
print('num_updates:', ck.get('num_updates'))
print('Has optimizer_state_dict:', 'optimizer_state_dict' in ck)

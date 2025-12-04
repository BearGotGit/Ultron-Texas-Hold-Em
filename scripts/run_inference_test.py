import sys, pathlib, torch, glob, os

# Ensure project root is on sys.path so package imports (e.g. `training`) work
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add placeholder PPOConfig in __main__ for safe unpickling if needed
try:
    import __main__ as _m
    if not hasattr(_m, 'PPOConfig'):
        class PPOConfig:
            pass
        setattr(_m, 'PPOConfig', PPOConfig)
except Exception:
    pass

print('Using cwd:', os.getcwd())

# Find latest rollout
rollouts = sorted(glob.glob('rollouts/*.pt'))
if not rollouts:
    raise SystemExit('No rollout files found in rollouts/')
latest = rollouts[-1]
print('Loading rollout:', latest)

# Try weights-only first
try:
    r = torch.load(latest, map_location='cpu', weights_only=True)
    print('Loaded rollout with weights_only=True')
except Exception:
    r = torch.load(latest, map_location='cpu', weights_only=False)
    print('Loaded rollout with full load')

if 'observations' not in r:
    raise SystemExit('Rollout does not contain observations key')
obs = r['observations']
print('Observations tensor shape:', getattr(obs, 'shape', str(type(obs))))

# Flatten to pick a single observation sample
if isinstance(obs, torch.Tensor):
    if obs.ndim == 3:
        sample = obs[0, 0]
    elif obs.ndim == 2:
        sample = obs[0]
    else:
        sample = obs.reshape(-1, obs.shape[-1])[0]
else:
    raise SystemExit('Observations not a tensor')

print('Sample observation shape:', sample.shape)

# Load model class
from training.ppo_model import PokerPPOModel

# Load checkpoint
ckpt_path = 'checkpoints/final.pt'
if not os.path.exists(ckpt_path):
    raise SystemExit('Checkpoint not found: ' + ckpt_path)

# try weights-only load for checkpoint
try:
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    print('Loaded checkpoint weights_only')
    # If weights_only returns a dict of tensors (state_dict)
    state_dict = ck
except Exception:
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('Loaded full checkpoint')
    state_dict = ck.get('model_state_dict', ck.get('state_dict', None))

if state_dict is None:
    raise SystemExit('Could not find model_state_dict in checkpoint')

# Build model and load state
model = PokerPPOModel()
try:
    model.load_state_dict(state_dict)
    print('Loaded model state_dict into PokerPPOModel')
except Exception as e:
    # If state_dict contains nested keys like 'module.' prefixes, try to adapt
    new_state = {}
    for k,v in state_dict.items():
        new_k = k.replace('module.', '')
        new_state[new_k] = v
    model.load_state_dict(new_state)
    print('Loaded model state_dict after stripping module. prefixes')

model.eval()
with torch.no_grad():
    sample_tensor = torch.as_tensor(sample).unsqueeze(0).float()
    action, logp, entropy, value = model.get_action_and_value(sample_tensor, deterministic=True)

print('Deterministic action (p_fold, bet_scalar):', action.squeeze(0).cpu().numpy())
print('Log prob:', logp.squeeze(0).cpu().numpy())
print('Value:', value.squeeze(0).cpu().numpy())

print('\n✓ Inference smoke test passed')

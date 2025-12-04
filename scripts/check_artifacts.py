import torch, glob, os, traceback

ck = 'checkpoints/final.pt'
print('CKPT_EXISTS', os.path.exists(ck))
# Some checkpoints were saved when `PPOConfig` was defined in `__main__`.
# Provide a minimal placeholder class under `__main__` so unpickling can find it.
try:
    import __main__ as _m
    if not hasattr(_m, 'PPOConfig'):
        class PPOConfig:  # placeholder used only for unpickling
            pass
        setattr(_m, 'PPOConfig', PPOConfig)
except Exception:
    pass
# Try a weights-only load first (safer): avoids unpickling custom classes
try:
    ckpt = torch.load(ck, map_location='cpu', weights_only=True)
    print('CKPT_TYPE_weights_only', type(ckpt))
    # weights-only load often returns a dict of tensors (e.g., state_dict)
    try:
        print('CKPT_KEYS_weights_only', list(ckpt.keys())[:50])
    except Exception:
        print('CKPT_WEIGHTS_ONLY_CONTENT', str(type(ckpt)))
except Exception:
    print('CKPT_weights_only_failed, attempting full load')
    try:
        # Full load may require allowing globals; do a best-effort full load
        ckpt = torch.load(ck, map_location='cpu', weights_only=False)
        print('CKPT_TYPE_full', type(ckpt))
        print('CKPT_KEYS_full', list(ckpt.keys())[:50])
        sd = ckpt.get('state_dict', None)
        print('HAS_state_dict', sd is not None)
        if sd is not None:
            print('NUM_PARAMS', len(sd))
    except Exception:
        print('CKPT_FULL_LOAD_ERR')
        traceback.print_exc()

rls = sorted(glob.glob('rollouts/*.pt'))
print('NUM_ROLLOUTS', len(rls))
if rls:
    r = rls[-1]
    try:
        d = torch.load(r, map_location='cpu', weights_only=False)
        print('ROLLOUT', r)
        print('ROLLOUT_KEYS', list(d.keys())[:50])
        print('SAMPLED_KEY_TYPES', {k:(getattr(v,'shape',str(type(v))) if hasattr(v,'shape') or hasattr(v,'size') else str(type(v))) for k,v in list(d.items())[:10]})
    except Exception:
        print('ROLLOUT_LOAD_ERR')
        traceback.print_exc()

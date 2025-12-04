"""Analyze rollouts to produce action-distribution and bet-size statistics.

Usage:
    python scripts/rollout_audit.py --pattern "rollouts/rollout_rl_from_bc_*.pt" --out stats.json
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from collections import defaultdict, Counter

# Ensure repo imports work
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.poker_env import interpret_action
from agents.poker_player import ActionType


def decode_normalized_log(value, normalizer):
    # value = log1p(x)/normalizer -> x = exp(value * normalizer) - 1
    return float(np.expm1(value * normalizer))


def analyze_rollouts(files, out=None, max_files=None):
    totals = Counter()
    per_stage = defaultdict(Counter)
    bet_scalars = defaultdict(list)
    bet_amounts = defaultdict(list)

    files = sorted(files)
    if max_files:
        files = files[:max_files]

    import __main__ as _m
    if not hasattr(_m, 'PPOConfig'):
        class PPOConfig:
            pass
        setattr(_m, 'PPOConfig', PPOConfig)

    for f in files:
        ck = torch.load(str(f), map_location='cpu', weights_only=False)
        obs_tensor = ck.get('observations')
        actions_tensor = ck.get('actions')
        cfg = ck.get('config', None)

        if obs_tensor is None or actions_tensor is None:
            continue

        obs = obs_tensor.numpy()
        acts = actions_tensor.numpy()

        # Get normalizers from config if present
        big_blind = getattr(cfg, 'big_blind', 10)
        starting_stack = getattr(cfg, 'starting_stack', 1000)
        bb_normalizer = np.log1p(big_blind)
        stack_normalizer = np.log1p(starting_stack)

        # obs shape: (num_steps, num_envs, obs_dim)
        ns, ne, od = obs.shape
        for t in range(ns):
            for e in range(ne):
                o = obs[t, e]
                a = acts[t, e]
                # Extract round stage from obs index 420 (0-based)
                # Global features start at index obs_dim - 6
                # round_stage normalized stored at global_features[3] -> obs[idx]
                # Compute indices dynamically
                global_start = od - 6
                round_norm = o[global_start + 3]
                # map back to 0..4
                round_stage_idx = int(round(round_norm * 4))
                stage = ['pre-flop', 'flop', 'turn', 'river', 'showdown'][max(0, min(4, round_stage_idx))]

                # Decode numeric globals
                current_bet = decode_normalized_log(o[global_start + 1], bb_normalizer)
                min_raise = decode_normalized_log(o[global_start + 2], bb_normalizer)

                # Hero player features at player block 0
                player_start = global_start - (9 * 4)
                hero_money_norm = o[player_start + 0]
                hero_bet_norm = o[player_start + 1]
                hero_money = decode_normalized_log(hero_money_norm, stack_normalizer)
                hero_bet = decode_normalized_log(hero_bet_norm, bb_normalizer)

                p_fold = float(a[0])
                bet_scalar = float(a[1])

                poker_action = interpret_action(
                    p_fold=p_fold,
                    bet_scalar=bet_scalar,
                    current_bet=int(round(current_bet)),
                    my_bet=int(round(hero_bet)),
                    min_raise=int(round(min_raise)),
                    my_money=int(round(hero_money)),
                )

                atype = poker_action.action_type
                totals[atype] += 1
                per_stage[stage][atype] += 1

                # Record bet_scalar and amount for raises
                if atype == ActionType.RAISE:
                    bet_scalars['overall'].append(bet_scalar)
                    bet_scalars[stage].append(bet_scalar)
                    bet_amounts['overall'].append(poker_action.amount)
                    bet_amounts[stage].append(poker_action.amount)

    # Summarize
    summary = {
        'total_actions': sum(totals.values()),
        'action_counts': {k.name: int(v) for k, v in totals.items()},
        'per_stage_counts': {
            stage: {k.name: int(v) for k, v in counter.items()} for stage, counter in per_stage.items()
        },
        'bet_scalar_stats': {},
        'bet_amount_stats': {},
    }

    def stats_from_list(lst):
        if not lst:
            return None
        arr = np.array(lst)
        return {
            'count': int(arr.size),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
        }

    for key, lst in bet_scalars.items():
        summary['bet_scalar_stats'][key] = stats_from_list(lst)
    for key, lst in bet_amounts.items():
        summary['bet_amount_stats'][key] = stats_from_list(lst)

    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w') as fh:
            json.dump(summary, fh, indent=2)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='rollouts/rollout_rl_from_bc_*.pt')
    parser.add_argument('--out', type=str, default='artifacts/rollout_audit.json')
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    files = list(Path('.').glob(args.pattern))
    if not files:
        print('No rollout files found for pattern', args.pattern)
        raise SystemExit(1)

    summary = analyze_rollouts(files, out=args.out, max_files=args.max_files)
    print('Audit complete. Summary:')
    print(json.dumps(summary, indent=2))
    print('Saved to', args.out)

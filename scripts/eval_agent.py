import sys, pathlib, os, argparse, time
# Ensure repo root on path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import MonteCarloAgent
from agents.monte_carlo_agent import RandomAgent
from agents.poker_player import PokerPlayer, PokerPlayerPublic, PokerAction, ActionType
from training.ppo_model import PokerPPOModel

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pt')
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--opp-sims', type=int, default=200, help='MonteCarloAgent num_simulations')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load model; prefer RLAgent loader when available
if not os.path.exists(args.checkpoint):
    raise SystemExit('Checkpoint not found: ' + args.checkpoint)

# Provide placeholder PPOConfig on __main__ so torch can unpickle older checkpoints
try:
    import __main__ as _m
    if not hasattr(_m, 'PPOConfig'):
        class PPOConfig:
            pass
        setattr(_m, 'PPOConfig', PPOConfig)
except Exception:
    pass

try:
    from agents.rl_agent import RLAgent
except Exception:
    RLAgent = None

model = None
if RLAgent is not None:
    try:
        agent = RLAgent.from_checkpoint(args.checkpoint, player_id='eval-agent', starting_money=1000, device=args.device)
        model = agent.model
        model.to(args.device)
        model.eval()
        print('Loaded model via RLAgent.from_checkpoint')
    except Exception:
        model = None

if model is None:
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ck.get('model_state_dict', ck.get('state_dict', None))
    if state is None:
        raise SystemExit('No model_state_dict found in checkpoint')
    model = PokerPPOModel()
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

# Evaluation loop
episodes = args.episodes
wins = 0
profits = []
start_stack = 1000

print(f"Running evaluation: {episodes} episodes vs MonteCarloAgent")
start_time = time.time()
for ep in range(episodes):
    # Create players: hero placeholder (RandomAgent) and opponent MonteCarloAgent
    hero = RandomAgent('HERO', starting_money=start_stack)
    opp = MonteCarloAgent('OPP', starting_money=start_stack, num_simulations=args.opp_sims)
    players = [hero, opp]
    env = PokerEnv(players=players, config=PokerEnvConfig(starting_stack=start_stack), hero_idx=0)

    obs, info = env.reset()
    terminated = False
    total_reward = 0.0
    # Step loop
    while not terminated:
        # obs is numpy array for hero
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(args.device)
        with torch.no_grad():
            fold_logit, bet_alpha, bet_beta, _ = model(obs_tensor)
            p_fold = float(torch.sigmoid(fold_logit).squeeze().cpu().numpy())
            hero = RandomAgent('HERO', starting_money=start_stack)
            bet_alpha_v = float(bet_alpha.squeeze().cpu().numpy())
            bet_beta_v = float(bet_beta.squeeze().cpu().numpy())
            # Use mean for deterministic evaluation: alpha/(alpha+beta)
            bet_scalar = bet_alpha_v / (bet_alpha_v + bet_beta_v)
        hero_player = env.players[env.hero_idx]

        # compute amount to call
        to_call = max(0, env.current_bet - hero_player.bet)

        # convert RL output to safe PokerAction (identical to RLAgent behavior)
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=env.current_bet,
            my_bet=hero_player.bet,
            min_raise=max(env.min_raise, env.config.big_blind),
            my_money=hero_player.money,
        )

        # safety: folding with nothing to call → check
        if poker_action.action_type == ActionType.FOLD and to_call == 0:
            poker_action = PokerAction.check()

        # safety: raise smaller than call → convert to call
        if poker_action.action_type == ActionType.RAISE:
            if poker_action.amount < to_call:
                poker_action = PokerAction.call(to_call)


        # If a trained model is loaded, the env expects a numeric action array
        # (`[p_fold, bet_scalar]`) so let the env handle discretization.
        if model is not None:
            action_arr = np.array([p_fold, bet_scalar], dtype=float)
            obs, reward, terminated, truncated, info = env.step(action_arr)
        else:
            obs, reward, terminated, truncated, info = env.step(poker_action)

        total_reward += reward
        if truncated:
            # treat truncated as end
            break
    final_chips = env.players[0].money
    profit = final_chips - start_stack
    profits.append(profit)
    if profit > 0:
        wins += 1
    if (ep + 1) % 10 == 0 or ep == episodes - 1:
        print(f"  Episode {ep+1}/{episodes} - profit={profit}, total_reward={total_reward:.4f}")

elapsed = time.time() - start_time
win_rate = wins / episodes * 100.0
avg_profit = float(np.mean(profits))
print('\nEvaluation complete')
print(f'  Episodes: {episodes}')
print(f'  Win rate: {win_rate:.2f}%')
print(f'  Avg profit (chips): {avg_profit:.2f}')
print(f'  Time: {elapsed:.1f}s')

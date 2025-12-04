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

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pt')
parser.add_argument('--episodes', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--render', action='store_true')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load checkpoint with PPOConfig placeholder
try:
    import __main__ as _m
    if not hasattr(_m, 'PPOConfig'):
        class PPOConfig:
            pass
        setattr(_m, 'PPOConfig', PPOConfig)
except Exception:
    pass

if not os.path.exists(args.checkpoint):
    raise SystemExit('Checkpoint not found: ' + args.checkpoint)

# Prefer canonical RLAgent loader when available, otherwise fall back
try:
    from agents.rl_agent import RLAgent
except Exception:
    RLAgent = None

model = None
if RLAgent is not None:
    try:
        agent = RLAgent.from_checkpoint(args.checkpoint, player_id='script-rl', starting_money=1000, device=args.device)
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
        raise SystemExit('No model_state_dict in checkpoint')
    from training.ppo_model import PokerPPOModel
    model = PokerPPOModel()
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

start_stack = 1000

print(f"Playing {args.episodes} hands: RL (hero) vs MonteCarloAgent")

# Aggregate stats
wins = 0
total_profit = 0
total_final_chips = 0

for ep in range(args.episodes):
    # Create players placeholders: hero (RandomAgent used as state holder), opponent MonteCarloAgent
    hero = RandomAgent('HERO', starting_money=start_stack)
    opp = MonteCarloAgent('OPP', starting_money=start_stack, num_simulations=200)
    players = [hero, opp]
    env = PokerEnv(players=players, config=PokerEnvConfig(starting_stack=start_stack), hero_idx=0, render_mode='human' if args.render else None)

    obs, info = env.reset()
    terminated = False
    step_idx = 0
    if args.render:
        env.render()
    while not terminated:
        # Model acts only when it's hero's turn; env will advance opponents internally
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(args.device)
        with torch.no_grad():
            fold_logit, bet_alpha, bet_beta, _ = model(obs_tensor)
            p_fold = float(torch.sigmoid(fold_logit).squeeze().cpu().numpy())
            bet_alpha_v = float(bet_alpha.squeeze().cpu().numpy())
            bet_beta_v = float(bet_beta.squeeze().cpu().numpy())
            bet_scalar = bet_alpha_v / (bet_alpha_v + bet_beta_v)
        action_arr = np.array([p_fold, bet_scalar], dtype=np.float32)

        # Print the model's raw outputs and interpreted action
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=env.current_bet,
            my_bet=hero.bet,
            min_raise=env.min_raise,
            my_money=hero.money,
        )
        print(f"Episode {ep+1} Step {step_idx}: Model outputs p_fold={p_fold:.3f}, bet_scalar={bet_scalar:.3f} -> Interpreted: {poker_action.action_type.value}, amount={poker_action.amount}")

        obs, reward, terminated, truncated, info = env.step(action_arr)
        if args.render:
            env.render()
            time.sleep(0.2)
        step_idx += 1
        if truncated:
            print('Episode truncated')
            break

    final_chips = env.players[0].money
    profit = final_chips - start_stack
    print(f"Episode {ep+1} finished: hero chips={final_chips}, profit={profit}\n")

    # Update aggregates
    total_final_chips += final_chips
    total_profit += profit
    if profit > 0:
        wins += 1

# Print summary
episodes_ran = args.episodes if args.episodes > 0 else 1
win_rate = wins / episodes_ran
avg_profit = total_profit / episodes_ran
avg_final_chips = total_final_chips / episodes_ran

print('--- Summary ---')
print(f'Episodes: {episodes_ran}')
print(f'Win rate: {win_rate:.2%} ({wins}/{episodes_ran})')
print(f'Avg profit (chips): {avg_profit:.2f}')
print(f'Avg final chips: {avg_final_chips:.2f}')
print('Done')

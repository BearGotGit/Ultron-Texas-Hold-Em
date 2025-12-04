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

print('Done')

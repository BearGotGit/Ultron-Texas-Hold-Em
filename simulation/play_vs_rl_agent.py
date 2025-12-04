"""
Interactive game: Human vs Trained RL Agent

Play Texas Hold'em against your trained PPO model.
Reuses the PokerEnv simulator for game flow.
"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path

from treys import Card
from simulation.poker_env import PokerEnv, PokerEnvConfig
from training.ppo_model import PokerPPOModel
from training.train_rl_model import PPOConfig
from agents.poker_player import PokerPlayer, PokerAction, ActionType
from agents.human_player import HumanPlayer
from utils.device import DEVICE


class RLAgentPlayer(PokerPlayer):
    """Wrapper for trained RL model."""
    
    def __init__(
        self,
        player_id: str,
        starting_money: int,
        model: PokerPPOModel,
        env: PokerEnv,
        device: torch.device,
    ):
        super().__init__(player_id, starting_money)
        self.model = model
        self.env = env
        self.device = device
    
    def get_action(
        self,
        hole_cards,
        board,
        pot,
        current_bet,
        min_raise,
        players,
        my_idx,
    ) -> PokerAction:
        """Get action from trained RL model."""
        # Get observation from environment
        obs = self.env._get_observation()
        
        # Convert to tensor
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get deterministic action from model
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
        
        action_np = action.squeeze(0).cpu().numpy()
        
        # Interpret action using environment's interpreter
        from simulation.poker_env import interpret_action
        poker_action = interpret_action(
            p_fold=float(action_np[0]),
            bet_scalar=float(action_np[1]),
            current_bet=current_bet,
            my_bet=self.bet,
            min_raise=min_raise,
            my_money=self.money,
        )
        
        # Print what RL agent is doing
        print(f"\n🤖 {self.id} action: {poker_action.action_type.value.upper()}", end="")
        if poker_action.amount > 0:
            print(f" ${poker_action.amount}")
        else:
            print()
        
        return poker_action


def play_game(
    model_path: str,
    num_hands: int = 10,
    starting_stack: int = 1000,
    big_blind: int = 10,
):
    """
    Play poker against a trained RL model.
    
    Args:
        model_path: Path to trained model checkpoint
        num_hands: Number of hands to play
        starting_stack: Starting chips for each player
        big_blind: Big blind amount
    """
    print("\n" + "="*70)
    print("🃏 TEXAS HOLD'EM: HUMAN vs RL AGENT")
    print("="*70)
    
    # Load model
    device = DEVICE
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = PokerPPOModel(
        card_embed_dim=64,
        hidden_dim=256,
        num_shared_layers=2,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['global_step']:,} steps)")
    
    # Create players
    human = HumanPlayer("Human", starting_stack)
    
    # Create environment config
    config = PokerEnvConfig(
        big_blind=big_blind,
        small_blind=big_blind // 2,
        starting_stack=starting_stack,
        max_players=2,
    )
    
    # Create RL agent wrapper  
    # We'll create two envs - one for each player as "hero"
    from agents.monte_carlo_agent import RandomAgent
    
    # Env 1: Human is player 0, RL is player 1 (hero)
    placeholder_human = RandomAgent("Human", starting_stack)
    placeholder_rl = RandomAgent("RL-Agent", starting_stack)
    
    env = PokerEnv(
        players=[placeholder_human, placeholder_rl],
        config=config,
        hero_idx=1,  # RL agent is hero
    )
    
    # Create actual RL agent wrapper
    rl_agent = RLAgentPlayer("RL-Agent", starting_stack, model, env, device)
    
    # Replace placeholders
    env.players[0] = human
    env.players[1] = rl_agent
    
    print(f"\nGame settings:")
    print(f"  Starting stack: ${starting_stack}")
    print(f"  Blinds: ${config.small_blind}/${config.big_blind}")
    print(f"  Hands to play: {num_hands}")
    
    # Play hands
    for hand_num in range(1, num_hands + 1):
        print(f"\n\n{'#'*70}")
        print(f"# HAND {hand_num}/{num_hands}")
        print(f"{'#'*70}")
        print(f"Human: ${human.money} | RL-Agent: ${rl_agent.money}")
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        # Game loop - let environment handle flow
        while not (terminated or truncated):
            # It's RL agent's turn (hero)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                fold_logit, bet_alpha, bet_beta, value = model(obs_tensor)
                
                # Sample action
                p_fold = torch.sigmoid(fold_logit).item()
                bet_dist = torch.distributions.Beta(bet_alpha, bet_beta)
                bet_scalar = bet_dist.sample().item()
                
                action = np.array([p_fold, bet_scalar], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Show showdown
        print(f"\n{'='*70}")
        print("SHOWDOWN")
        print(f"{'='*70}")
        
        # Show final board
        if env.board:
            print("\nFinal Board:")
            print("  " + "  ".join([Card.int_to_pretty_str(c) for c in env.board]))
        
        # Show all players' cards
        for player in env.players:
            cards_str = "  ".join([Card.int_to_pretty_str(c) for c in player.get_hole_cards()])
            status = " [FOLDED]" if player.folded else ""
            print(f"\n{player.id}:{status}")
            print(f"  {cards_str}")
        
        # Show winner info from environment
        if 'winner' in info:
            winner_idx = info['winner']
            winner = env.players[winner_idx]
            print(f"\n🏆 Winner: {winner.id}")
            if 'hand_name' in info:
                print(f"   Hand: {info['hand_name']}")
        
        # Show hand result
        print(f"\n{'='*70}")
        print(f"Hand {hand_num} complete!")
        print(f"Human: ${human.money} | RL-Agent: ${rl_agent.money}")
        
        # Check if anyone is out of chips
        if human.money <= 0:
            print("\n💔 You're out of chips! RL Agent wins the match!")
            break
        if rl_agent.money <= 0:
            print("\n🎉 RL Agent is out of chips! You win the match!")
            break
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("GAME SUMMARY")
    print(f"{'='*70}")
    print(f"Final stacks:")
    print(f"  Human: ${human.money} ({human.money - starting_stack:+d})")
    print(f"  RL-Agent: ${rl_agent.money} ({rl_agent.money - starting_stack:+d})")
    
    if human.money > rl_agent.money:
        print("\n🎉 You win!")
    elif rl_agent.money > human.money:
        print("\n🤖 RL Agent wins!")
    else:
        print("\n🤝 It's a tie!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Play Texas Hold'em against trained RL agent")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--stack", type=int, default=1000, help="Starting stack size")
    parser.add_argument("--big-blind", type=int, default=10, help="Big blind amount")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.rglob("*.pt")):
                print(f"  {ckpt}")
        return
    
    play_game(
        model_path=str(model_path),
        num_hands=args.hands,
        starting_stack=args.stack,
        big_blind=args.big_blind,
    )


if __name__ == "__main__":
    main()

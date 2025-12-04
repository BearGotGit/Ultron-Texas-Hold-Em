"""
Play Texas Hold'em against a trained RL agent.
Uses the existing TexasHoldemSimulation game engine.
"""

import argparse
from pathlib import Path
import torch

from agents.human_player import HumanPlayer
from agents.rl_agent import RLAgent
from simulation.poker_simulator import TexasHoldemSimulation
from utils.device import DEVICE


def play_vs_rl(
    model_path: str,
    num_hands: int = 10,
    starting_stack: int = 1000,
    big_blind: int = 10,
):
    """
    Play poker against a trained RL agent.
    
    Args:
        model_path: Path to checkpoint
        num_hands: Number of hands to play
        starting_stack: Starting chip stack
        big_blind: Big blind amount
    """
    print("="*70)
    print("🃏 TEXAS HOLD'EM: HUMAN vs RL AGENT")
    print("="*70)
    
    # Load model using RLAgent.from_checkpoint
    print(f"\nLoading model from {model_path}...")
    rl_agent = RLAgent.from_checkpoint(
        checkpoint_path=model_path,
        player_id="RL-Agent",
        starting_money=starting_stack,
        device=DEVICE,
    )
    print(f"✓ Model loaded")
    
    # Create human player
    human = HumanPlayer(name="You", starting_chips=starting_stack)
    
    agents = [human, rl_agent]
    
    # Create game
    small_blind = big_blind // 2
    game = TexasHoldemSimulation(agents, small_blind=small_blind, big_blind=big_blind)
    
    print(f"\nGame settings:")
    print(f"  Starting stack: ${starting_stack}")
    print(f"  Blinds: ${small_blind}/${big_blind}")
    print(f"  Hands to play: {num_hands}")
    
    # Play hands
    for hand_num in range(1, num_hands + 1):
        # Check if anyone is out
        if human.get_chips() <= 0:
            print("\n💔 You're out of chips! RL Agent wins!")
            break
        if rl_agent.get_chips() <= 0:
            print("\n🎉 RL Agent is out of chips! You win!")
            break
        
        print(f"\n\n{'#'*70}")
        print(f"# HAND {hand_num}/{num_hands}")
        print(f"{'#'*70}")
        print(f"You: ${human.get_chips()} | RL-Agent: ${rl_agent.get_chips()}")
        
        # Rotate dealer
        game.dealer_position = (game.dealer_position + 1) % len(agents)
        
        # Reset for new hand
        game.reset_for_new_hand()
        
        # Post blinds
        print(f"\n{'='*70}")
        print("POSTING BLINDS")
        print(f"{'='*70}")
        sb_pos, bb_pos = game.post_blinds()
        print(f"{agents[sb_pos]} posts small blind: ${game.small_blind}")
        print(f"{agents[bb_pos]} posts big blind: ${game.big_blind}")
        print(f"Pot: ${game.get_pot_size()}\n")
        
        # Deal hole cards
        game.deal_hole_cards()
        
        # Pre-flop
        game.run_betting_round("Pre-flop")
        
        # Check if hand over
        active = [a for a in agents if not a.is_folded]
        if len(active) <= 1:
            if active:
                print(f"\n{active[0]} wins ${game.get_pot_size()} (opponent folded)")
                active[0].add_chips(game.get_pot_size())
            continue
        
        # Flop
        print(f"\n{'='*70}")
        print("DEALING THE FLOP")
        print(f"{'='*70}")
        game.deal_flop()
        game.print_board()
        print()
        
        game.run_betting_round("Flop")
        
        active = [a for a in agents if not a.is_folded]
        if len(active) <= 1:
            if active:
                print(f"\n{active[0]} wins ${game.get_pot_size()} (opponent folded)")
                active[0].add_chips(game.get_pot_size())
            continue
        
        # Turn
        print(f"\n{'='*70}")
        print("DEALING THE TURN")
        print(f"{'='*70}")
        game.deal_turn()
        game.print_board()
        print()
        
        game.run_betting_round("Turn")
        
        active = [a for a in agents if not a.is_folded]
        if len(active) <= 1:
            if active:
                print(f"\n{active[0]} wins ${game.get_pot_size()} (opponent folded)")
                active[0].add_chips(game.get_pot_size())
            continue
        
        # River
        print(f"\n{'='*70}")
        print("DEALING THE RIVER")
        print(f"{'='*70}")
        game.deal_river()
        game.print_board()
        print()
        
        game.run_betting_round("River")
        
        # Showdown
        print(f"\n{'='*70}")
        print("SHOWDOWN")
        print(f"{'='*70}\n")
        
        from treys import Card
        results = game.evaluate_hands()
        for agent, score, hand_name, percentage in results:
            if not agent.is_folded:
                print(f"{agent}:")
                print(f"  Cards: ", end="")
                Card.print_pretty_cards(agent.get_hole_cards())
                print(f"  Hand: {hand_name}")
                print()
        
        # Award pot
        print(f"{'='*70}")
        print(f"Total pot: ${game.get_pot_size()}\n")
        
        winnings = game.award_pot()
        for winner, amount, pot_desc in winnings:
            print(f"{winner} wins ${amount}! ({pot_desc})")
        
        print(f"\n{'='*70}")
        print(f"Hand {hand_num} complete!")
        print(f"You: ${human.get_chips()} | RL-Agent: ${rl_agent.get_chips()}")
        print(f"{'='*70}")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("GAME SUMMARY")
    print(f"{'='*70}")
    print(f"Final stacks:")
    print(f"  You: ${human.get_chips()} ({human.get_chips() - starting_stack:+d})")
    print(f"  RL-Agent: ${rl_agent.get_chips()} ({rl_agent.get_chips() - starting_stack:+d})")
    
    if human.get_chips() > rl_agent.get_chips():
        print("\n🎉 You win!")
    elif rl_agent.get_chips() > human.get_chips():
        print("\n🤖 RL Agent wins!")
    else:
        print("\n🤝 It's a tie!")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Play poker against trained RL agent")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--stack", type=int, default=1000, help="Starting chip stack")
    parser.add_argument("--big-blind", type=int, default=10, help="Big blind amount")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    play_vs_rl(
        model_path=str(model_path),
        num_hands=args.hands,
        starting_stack=args.stack,
        big_blind=args.big_blind,
    )


if __name__ == "__main__":
    main()

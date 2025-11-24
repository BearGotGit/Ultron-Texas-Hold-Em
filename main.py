"""
Main entry point for Texas Hold'em poker simulation.
Demonstrates usage of PokerAgent and TexasHoldemSimulation with betting.
"""

from treys import Card
from agents import PokerAgent
from agents.human_player import HumanPlayer
from simulation import TexasHoldemSimulation


def run_single_hand(agents, game, hand_number=1, show_equity=True):
    """
    Run a single hand of poker with existing agents.
    
    Args:
        agents: List of PokerAgent instances (with existing chip stacks)
        game: TexasHoldemSimulation instance
        hand_number: Hand number for display (default: 1)
        show_equity: Whether to show equity calculations (default: True)
    """
    print(f"\n{'='*60}")
    print(f"HAND #{hand_number}")
    print(f"{'='*60}\n")
    
    # Reset game for new hand
    game.reset_for_new_hand()
    
    # Show chip stacks
    print("Current chip stacks:")
    for agent in agents:
        print(f"  {agent}: ${agent.get_chips()}")
    print()
    
    # Post blinds
    print(f"{'='*60}")
    print("POSTING BLINDS")
    print(f"{'='*60}")
    sb_pos, bb_pos = game.post_blinds()
    print(f"{agents[sb_pos]} posts small blind: ${game.small_blind}")
    print(f"{agents[bb_pos]} posts big blind: ${game.big_blind}")
    print(f"Pot: ${game.get_pot_size()}")
    print()
    
    # Deal hole cards
    game.deal_hole_cards()
    print(f"{'='*60}")
    print("HOLE CARDS DEALT")
    print(f"{'='*60}\n")
    for agent in agents:
        if agent.get_chips() > 0:  # Only show active players
            print(f"{agent} (${agent.get_chips()}):")
            Card.print_pretty_cards(agent.get_hole_cards())
            print()
    
    # Pre-flop equity
    if show_equity:
        print(f"{'='*60}")
        print("PRE-FLOP EQUITY")
        print(f"{'='*60}")
        preflop_equity = game.calculate_all_equities()
        for i, (agent, equity) in enumerate(zip(agents, preflop_equity)):
            if agent.get_chips() > 0:
                print(f"{agent}: {equity:.2%} equity")
        print()
    
    # Pre-flop betting
    game.run_betting_round("Pre-flop")
    
    # Check if hand is over
    active_players = [a for a in agents if not a.is_folded and a.get_chips() >= 0]
    if len(active_players) <= 1:
        if active_players:
            print(f"\n{active_players[0]} wins ${game.get_pot_size()} (all others folded)")
            active_players[0].add_chips(game.get_pot_size())
        return
    
    # Deal flop
    print(f"\n{'='*60}")
    print("DEALING THE FLOP")
    print(f"{'='*60}")
    game.deal_flop()
    game.print_board()
    print()
    
    # Flop equity
    if show_equity:
        print(f"{'='*60}")
        print("FLOP EQUITY")
        print(f"{'='*60}")
        flop_equity = game.calculate_all_equities()
        for i, (agent, equity) in enumerate(zip(agents, flop_equity)):
            if not agent.is_folded:
                change = flop_equity[i] - preflop_equity[i]
                print(f"{agent}: {equity:.2%} equity ({change:+.2%})")
        print()
    
    # Flop betting
    game.run_betting_round("Flop")
    
    # Check if hand is over
    active_players = [a for a in agents if not a.is_folded]
    if len(active_players) <= 1:
        if active_players:
            print(f"\n{active_players[0]} wins ${game.get_pot_size()} (all others folded)")
            active_players[0].add_chips(game.get_pot_size())
        return
    
    # Deal turn
    print(f"\n{'='*60}")
    print("DEALING THE TURN")
    print(f"{'='*60}")
    game.deal_turn()
    game.print_board()
    print()
    
    # Turn equity
    if show_equity:
        print(f"{'='*60}")
        print("TURN EQUITY")
        print(f"{'='*60}")
        turn_equity = game.calculate_all_equities()
        for i, (agent, equity) in enumerate(zip(agents, turn_equity)):
            if not agent.is_folded:
                change = turn_equity[i] - flop_equity[i]
                print(f"{agent}: {equity:.2%} equity ({change:+.2%})")
        print()
    
    # Turn betting
    game.run_betting_round("Turn")
    
    # Check if hand is over
    active_players = [a for a in agents if not a.is_folded]
    if len(active_players) <= 1:
        if active_players:
            print(f"\n{active_players[0]} wins ${game.get_pot_size()} (all others folded)")
            active_players[0].add_chips(game.get_pot_size())
        return
    
    # Deal river
    print(f"\n{'='*60}")
    print("DEALING THE RIVER")
    print(f"{'='*60}")
    game.deal_river()
    game.print_board()
    print()
    
    # Final equity
    if show_equity:
        print(f"{'='*60}")
        print("RIVER EQUITY (FINAL)")
        print(f"{'='*60}")
        river_equity = game.calculate_all_equities()
        for i, (agent, equity) in enumerate(zip(agents, river_equity)):
            if not agent.is_folded:
                change = river_equity[i] - turn_equity[i]
                print(f"{agent}: {equity:.2%} equity ({change:+.2%})")
        print()
    
    # River betting
    game.run_betting_round("River")
    
    # Hand evaluation
    print(f"\n{'='*60}")
    print("SHOWDOWN - HAND EVALUATION")
    print(f"{'='*60}\n")
    
    results = game.evaluate_hands()
    for agent, score, hand_name, percentage in results:
        if not agent.is_folded:
            print(f"{agent}:")
            print(f"  Cards: ", end="")
            Card.print_pretty_cards(agent.get_hole_cards())
            print(f"  Hand: {hand_name}")
            print(f"  Rank: {score} ({percentage:.2%})")
            print(f"  Invested this hand: ${agent.total_invested}")
            print()
    
    # Award pot and determine winner
    print(f"{'='*60}")
    print("WINNER")
    print(f"{'='*60}")
    print(f"Total pot: ${game.get_pot_size()}\n")
    
    winnings = game.award_pot()
    
    # Group winnings by pot
    pot_groups = {}
    for winner, amount, pot_desc in winnings:
        if pot_desc not in pot_groups:
            pot_groups[pot_desc] = []
        pot_groups[pot_desc].append((winner, amount))
    
    # Display each pot's results
    for pot_desc, pot_winners in pot_groups.items():
        print(f"{pot_desc}:")
        for winner, amount in pot_winners:
            print(f"  {winner} wins ${amount}!")
        print()
    
    print(f"{'='*60}\n")


def run_full_game(num_players=4, starting_chips=1000):
    """
    Run a single complete hand with new agents.
    
    Args:
        num_players: Number of players (default: 4)
        starting_chips: Starting chip stack for each player (default: 1000)
    """
    print(f"\n{'='*60}")
    print(f"TEXAS HOLD'EM SIMULATION - {num_players} Players")
    print(f"{'='*60}\n")
    
    # Create agents
    agents = [PokerAgent(name=f"Player {i+1}", starting_chips=starting_chips) for i in range(num_players)]
    
    # Create game simulation
    game = TexasHoldemSimulation(agents, small_blind=5, big_blind=10)
    
    # Run one hand
    run_single_hand(agents, game, hand_number=1, show_equity=True)
    
    # Show final chip counts
    print(f"{'='*60}")
    print("FINAL CHIP COUNTS")
    print(f"{'='*60}")
    for agent in agents:
        profit = agent.get_chips() - starting_chips
        print(f"{agent}: ${agent.get_chips()} ({profit:+d})")
    print(f"\n{'='*60}\n")


def run_tournament(num_players=4, starting_chips=1000, num_hands=10):
    """
    Run multiple hands with the same agents - chip stacks persist!
    
    Args:
        num_players: Number of players (default: 4)
        starting_chips: Starting chip stack for each player (default: 1000)
        num_hands: Number of hands to play (default: 10)
    """
    print(f"\n{'='*60}")
    print(f"TEXAS HOLD'EM TOURNAMENT")
    print(f"{num_players} Players - {num_hands} Hands")
    print(f"{'='*60}\n")
    
    # Create agents ONCE - they persist across hands
    agents = [PokerAgent(name=f"Player {i+1}", starting_chips=starting_chips) for i in range(num_players)]
    
    # Create game simulation ONCE - shares agents
    game = TexasHoldemSimulation(agents, small_blind=5, big_blind=10)
    
    # Run multiple hands
    for hand_num in range(1, num_hands + 1):
        # Check if any players are out of chips
        active_agents = [a for a in agents if a.get_chips() > 0]
        if len(active_agents) <= 1:
            print(f"\n{'='*60}")
            print("TOURNAMENT OVER - Only one player with chips!")
            print(f"{'='*60}\n")
            break
        
        # Rotate dealer position
        game.dealer_position = (game.dealer_position + 1) % len(agents)
        
        # Run the hand (equity shown only on first few hands to reduce output)
        run_single_hand(agents, game, hand_number=hand_num, show_equity=(hand_num <= 3))
        
        # Show standings after each hand
        print(f"{'='*60}")
        print(f"STANDINGS AFTER HAND {hand_num}")
        print(f"{'='*60}")
        sorted_agents = sorted(agents, key=lambda a: a.get_chips(), reverse=True)
        for i, agent in enumerate(sorted_agents, 1):
            profit = agent.get_chips() - starting_chips
            status = "💀 BUSTED" if agent.get_chips() == 0 else ""
            print(f"{i}. {agent}: ${agent.get_chips()} ({profit:+d}) {status}")
        print()
    
    # Final tournament results
    print(f"\n{'='*60}")
    print("TOURNAMENT FINAL RESULTS")
    print(f"{'='*60}")
    sorted_agents = sorted(agents, key=lambda a: a.get_chips(), reverse=True)
    for i, agent in enumerate(sorted_agents, 1):
        profit = agent.get_chips() - starting_chips
        print(f"{i}. {agent}: ${agent.get_chips()} ({profit:+d})")
    
    if sorted_agents[0].get_chips() > 0:
        print(f"\n🏆 {sorted_agents[0]} wins the tournament!")
    
    print(f"\n{'='*60}\n")


def play_interactive(num_opponents=3, starting_chips=1000, num_hands=10):
    """
    Play poker interactively against AI opponents.
    
    Args:
        num_opponents: Number of AI opponents (default: 3)
        starting_chips: Starting chip stack for each player (default: 1000)
        num_hands: Number of hands to play (default: 10)
    """
    print(f"\n{'='*60}")
    print(f"INTERACTIVE TEXAS HOLD'EM")
    print(f"You vs {num_opponents} AI Opponents - {num_hands} Hands")
    print(f"{'='*60}\n")
    
    # Create agents - Human player first, then AI opponents
    agents = [HumanPlayer(name="You", starting_chips=starting_chips)]
    agents.extend([PokerAgent(name=f"AI {i+1}", starting_chips=starting_chips) for i in range(num_opponents)])
    
    # Create game simulation
    game = TexasHoldemSimulation(agents, small_blind=5, big_blind=10)
    
    # Run multiple hands
    for hand_num in range(1, num_hands + 1):
        # Check if any players are out of chips
        active_agents = [a for a in agents if a.get_chips() > 0]
        if len(active_agents) <= 1:
            print(f"\n{'='*60}")
            print("GAME OVER - Only one player with chips!")
            print(f"{'='*60}\n")
            break
        
        # Check if human player is out
        if agents[0].get_chips() == 0:
            print(f"\n{'='*60}")
            print("You're out of chips! Game over.")
            print(f"{'='*60}\n")
            break
        
        # Rotate dealer position
        game.dealer_position = (game.dealer_position + 1) % len(agents)
        
        # Run the hand (show equity for human player)
        run_single_hand(agents, game, hand_number=hand_num, show_equity=False)
        
        # Show standings after each hand
        print(f"{'='*60}")
        print(f"STANDINGS AFTER HAND {hand_num}")
        print(f"{'='*60}")
        sorted_agents = sorted(agents, key=lambda a: a.get_chips(), reverse=True)
        for i, agent in enumerate(sorted_agents, 1):
            profit = agent.get_chips() - starting_chips
            status = "💀 BUSTED" if agent.get_chips() == 0 else ""
            print(f"{i}. {agent}: ${agent.get_chips()} ({profit:+d}) {status}")
        print()
        
        # Ask if user wants to continue
        if hand_num < num_hands:
            response = input("Continue to next hand? (y/n): ").strip().lower()
            if response != 'y':
                print("\nQuitting game...")
                break
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    sorted_agents = sorted(agents, key=lambda a: a.get_chips(), reverse=True)
    for i, agent in enumerate(sorted_agents, 1):
        profit = agent.get_chips() - starting_chips
        print(f"{i}. {agent}: ${agent.get_chips()} ({profit:+d})")
    
    human_rank = next(i for i, a in enumerate(sorted_agents, 1) if a.name == "You")
    if human_rank == 1:
        print(f"\n🏆 Congratulations! You won!")
    else:
        print(f"\nYou finished in {human_rank}{'st' if human_rank == 1 else 'nd' if human_rank == 2 else 'rd' if human_rank == 3 else 'th'} place.")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # INTERACTIVE MODE - Play against AI!
    play_interactive(num_opponents=3, starting_chips=1000, num_hands=10)
    
    # Run a single hand (AI only)
    # run_full_game(num_players=4, starting_chips=1000)
    
    # Run a multi-hand tournament where stacks persist (AI only)
    # run_tournament(num_players=4, starting_chips=1000, num_hands=40)
    
    # Customize tournament
    # run_tournament(num_players=6, starting_chips=500, num_hands=20)

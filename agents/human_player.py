"""
Human player class for interactive Texas Hold'em poker.
Allows a human to play against AI agents by prompting for decisions.
"""

from .agent import PokerAgent
from treys import Card


class HumanPlayer(PokerAgent):
    """
    Interactive human player that prompts for decisions via console input.
    Inherits from PokerAgent and overrides the make_decision method.
    """
    
    def __init__(self, name="Human", starting_chips=1000):
        """
        Initialize a human player.
        
        Args:
            name: Player name (default: "Human")
            starting_chips: Starting chip stack (default: 1000)
        """
        super().__init__(name=name, starting_chips=starting_chips)
    
    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Prompt the human player for their decision.
        
        Args:
            board: Current community cards
            pot_size: Current pot size
            current_bet_to_call: Amount needed to call
            min_raise: Minimum raise amount
            
        Returns:
            Tuple of (action, amount) where:
                action: 'fold', 'call', 'raise', 'check'
                amount: Amount to bet (0 for fold/check/call, raise amount for raise)
        """
        if self.is_folded or self.is_all_in:
            return ('check', 0)
        
        print(f"\n{'='*60}")
        print(f"YOUR TURN - {self.name}")
        print(f"{'='*60}")
        
        # Show player's hand
        print("Your hole cards:")
        Card.print_pretty_cards(self.hole_cards)
        print()
        
        # Show board if available
        if board:
            print("Board:")
            Card.print_pretty_cards(board)
            print()
        
        # Show game state
        print(f"Pot: ${pot_size}")
        print(f"Your chips: ${self.get_chips()}")
        print(f"Your current bet this round: ${self.current_bet}")
        print(f"Amount to call: ${current_bet_to_call}")
        
        if current_bet_to_call > 0:
            print(f"Minimum raise: ${min_raise}")
        
        # Show hand evaluation if board exists
        if len(board) >= 3:
            score, hand_class, hand_name, percentage = self.evaluate_hand(board)
            print(f"\nCurrent hand: {hand_name} (Rank: {score}, {percentage:.1%})")
        
        print()
        
        # Determine available actions
        available_actions = []
        
        if current_bet_to_call == 0:
            available_actions.append("check")
            available_actions.append("raise")
        else:
            available_actions.append("fold")
            if current_bet_to_call <= self.get_chips():
                available_actions.append("call")
            if current_bet_to_call < self.get_chips():
                available_actions.append("raise")
        
        # Get player input
        while True:
            print(f"Available actions: {', '.join(available_actions)}")
            action_input = input("Your action: ").strip().lower()
            
            if action_input not in available_actions:
                print(f"Invalid action. Please choose from: {', '.join(available_actions)}")
                continue
            
            # Handle check or fold
            if action_input == 'check':
                return ('check', 0)
            
            if action_input == 'fold':
                confirm = input("Are you sure you want to fold? (y/n): ").strip().lower()
                if confirm == 'y':
                    return ('fold', 0)
                else:
                    continue
            
            # Handle call
            if action_input == 'call':
                if current_bet_to_call >= self.get_chips():
                    confirm = input(f"Calling ${current_bet_to_call} will put you ALL-IN. Continue? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                return ('call', current_bet_to_call)
            
            # Handle raise
            if action_input == 'raise':
                max_raise = self.get_chips() - current_bet_to_call
                print(f"Minimum raise: ${min_raise}")
                print(f"Maximum raise: ${max_raise} (all-in)")
                
                while True:
                    try:
                        raise_amount = input(f"Raise amount (${min_raise}-${max_raise}): ").strip()
                        raise_amount = int(raise_amount)
                        
                        if raise_amount < min_raise:
                            print(f"Raise must be at least ${min_raise}")
                            continue
                        
                        if raise_amount > max_raise:
                            print(f"Raise cannot exceed ${max_raise}")
                            continue
                        
                        # Confirm all-in
                        if raise_amount == max_raise:
                            confirm = input(f"Raising ${raise_amount} will put you ALL-IN. Continue? (y/n): ").strip().lower()
                            if confirm != 'y':
                                break
                        
                        return ('raise', raise_amount)
                    
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\nAction cancelled. Choose a different action.")
                        break
        
        # Fallback (should never reach here)
        return ('fold', 0)
    
    def __str__(self):
        """String representation shows human player clearly."""
        return f"{self.name} 👤"

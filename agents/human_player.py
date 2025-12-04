"""
Human player class for interactive Texas Hold'em poker.
Allows a human to play against AI agents by prompting for decisions.

Inherits from PokerPlayer (new interface) but also provides backwards
compatibility with PokerAgent interface via make_decision adapter method.
"""

from typing import List
from treys import Card

from .poker_player import PokerPlayer, PokerAction, ActionType, PokerPlayerPublic


class HumanPlayer(PokerPlayer):
    """
    Interactive human player that prompts for decisions via console input.
    Inherits from PokerPlayer for the new interface.
    
    Provides backwards compatibility with PokerAgent-style calls via
    make_decision() method for use with TexasHoldemSimulation.
    """
    
    def __init__(self, player_id: str = "Human", starting_money: int = 1000, 
                 name: str = None, starting_chips: int = None):
        """
        Initialize a human player.
        
        Args:
            player_id: Unique identifier for this player (default: "Human")
            starting_money: Starting chip stack (default: 1000)
            name: Alias for player_id (for backwards compatibility)
            starting_chips: Alias for starting_money (for backwards compatibility)
        """
        # Handle backwards-compatible parameter names
        if name is not None:
            player_id = name
        if starting_chips is not None:
            starting_money = starting_chips
            
        super().__init__(player_id, starting_money)
        
        # For backwards compatibility with PokerAgent interface
        self.name = player_id
        self.hole_cards = []  # Alias for _private_cards
        self.chips = starting_money  # Alias for money
        self.current_bet = 0  # Tracks current bet in PokerAgent style
        self.total_invested = 0  # Total invested this hand
        self.is_folded = False  # Alias for folded
        self.is_all_in = False  # Alias for all_in
    
    # ================================================================
    # PokerPlayer Interface (new interface)
    # ================================================================
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        """
        Prompt human for action (PokerPlayer interface).
        
        Args:
            hole_cards: This player's hole cards (Treys integers)
            board: Community cards (Treys integers)
            pot: Current pot (int or Pot object)
            current_bet: Current bet to match
            min_raise: Minimum raise amount
            players: Public info for all players
            my_idx: This player's index in players list
            
        Returns:
            PokerAction describing the chosen action
        """
        print(f"\n{'='*70}")
        print(f"YOUR TURN - {self.id}")
        print(f"{'='*70}")
        
        # Show hole cards
        print("\nYour hole cards:")
        print("  " + "  ".join([Card.int_to_pretty_str(c) for c in hole_cards]))
        
        # Show board (with street label)
        if board:
            if len(board) == 3:
                street = "FLOP"
            elif len(board) == 4:
                street = "TURN"
            elif len(board) == 5:
                street = "RIVER"
            else:
                street = "BOARD"
            print(f"\n{street}:")
            print("  " + "  ".join([Card.int_to_pretty_str(c) for c in board]))
        else:
            print("\n(Pre-flop)")
        
        # Show game state
        pot_value = pot.money if hasattr(pot, 'money') else pot
        print(f"\n💰 Pot: ${pot_value}")
        print(f"💵 Your chips: ${self.money}")
        print(f"🎲 Your current bet: ${self.bet}")
        
        to_call = current_bet - self.bet
        print(f"📞 To call: ${to_call}")
        
        if to_call > 0:
            print(f"⬆️  Minimum raise: ${min_raise}")
        
        # Show other players
        print("\n👥 Other players:")
        for i, p in enumerate(players):
            if i == my_idx:
                continue
            status = []
            if p.folded:
                status.append("FOLDED")
            if p.all_in:
                status.append("ALL-IN")
            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"   {p.id}: ${p.money} (bet: ${p.bet}){status_str}")
        
        # Determine available actions
        print("\n" + "="*70)
        
        if to_call == 0:
            # Can check or raise
            print("Available actions:")
            print("  1. Check")
            print("  2. Raise")
            
            while True:
                choice = input("\nYour choice (1-2): ").strip()
                
                if choice == "1":
                    return PokerAction.check()
                elif choice == "2":
                    # Get raise amount
                    max_raise = self.money
                    while True:
                        try:
                            raise_amount = int(input(f"Raise amount (min ${min_raise}, max ${max_raise}): "))
                            if raise_amount < min_raise:
                                print(f"Minimum raise is ${min_raise}")
                                continue
                            if raise_amount > max_raise:
                                print(f"You only have ${max_raise}")
                                continue
                            return PokerAction.raise_to(raise_amount)
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    print("Invalid choice. Enter 1 or 2.")
        else:
            # Can fold, call, or raise
            print("Available actions:")
            print("  1. Fold")
            print(f"  2. Call ${to_call}")
            if self.money >= to_call + min_raise:
                print(f"  3. Raise")
            
            while True:
                choice = input("\nYour choice: ").strip()
                
                if choice == "1":
                    return PokerAction.fold()
                elif choice == "2":
                    return PokerAction.call(to_call)
                elif choice == "3" and self.money >= to_call + min_raise:
                    # Get raise amount
                    max_raise = self.money - to_call
                    while True:
                        try:
                            raise_amount = int(input(f"Raise amount (min ${min_raise}, max ${max_raise}): "))
                            if raise_amount < min_raise:
                                print(f"Minimum raise is ${min_raise}")
                                continue
                            if raise_amount > max_raise:
                                print(f"You only have ${max_raise} after calling")
                                continue
                            # Return just the raise amount (the call is implicit)
                            return PokerAction.raise_to(raise_amount)
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    print("Invalid choice.")
    
    # ================================================================
    # PokerAgent Interface (backwards compatibility)
    # ================================================================
    
    def receive_cards(self, cards: List[int]):
        """
        Receive hole cards (PokerAgent interface).
        
        Args:
            cards: List of 2 card integers
        """
        self.hole_cards = cards
        self._private_cards = cards
    
    def reset_for_new_hand(self):
        """Reset state for a new hand (PokerAgent interface)."""
        self.hole_cards = []
        self._private_cards = []
        self.current_bet = 0
        self.bet = 0
        self.total_invested = 0
        self.is_folded = False
        self.is_all_in = False
        self.folded = False
        self.all_in = False
    
    def reset_current_bet(self):
        """Reset current bet for a new betting round (PokerAgent interface)."""
        self.current_bet = 0
        self.bet = 0
    
    def get_hole_cards(self) -> List[int]:
        """Get hole cards (PokerAgent interface)."""
        return self.hole_cards if self.hole_cards else self._private_cards
    
    def get_chips(self) -> int:
        """Get current chip stack (PokerAgent interface)."""
        return self.money
    
    def add_chips(self, amount: int):
        """Add chips to stack (PokerAgent interface)."""
        self.money += amount
        self.chips = self.money
    
    def place_bet(self, amount: int) -> int:
        """
        Place a bet (PokerAgent interface).
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        if amount < 0:
            return 0
        
        actual_bet = min(amount, self.money)
        self.money -= actual_bet
        self.chips = self.money
        self.current_bet += actual_bet
        self.bet = self.current_bet
        self.total_invested += actual_bet
        
        if self.money == 0:
            self.is_all_in = True
            self.all_in = True
        
        return actual_bet
    
    def fold(self):
        """Fold the hand."""
        self.is_folded = True
        self.folded = True
    
    def evaluate_hand(self, board: List[int]):
        """
        Evaluate the strength of the current hand (PokerAgent interface).
        
        Args:
            board: List of community cards
            
        Returns:
            Tuple of (score, hand_class, hand_name, percentage)
        """
        from treys import Evaluator
        
        if len(board) < 3:
            return None, None, None, None
        
        evaluator = Evaluator()
        hole_cards = self.get_hole_cards()
        score = evaluator.evaluate(board, hole_cards)
        hand_class = evaluator.get_rank_class(score)
        hand_name = evaluator.class_to_string(hand_class)
        percentage = evaluator.get_five_card_rank_percentage(score)
        
        return score, hand_class, hand_name, percentage
    
    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Make a betting decision (PokerAgent interface for backwards compatibility).
        
        Adapts the PokerPlayer.get_action() interface to the PokerAgent interface
        used by TexasHoldemSimulation.
        
        Args:
            board: Current community cards
            pot_size: Current pot size
            current_bet_to_call: Amount needed to call
            min_raise: Minimum raise amount
            
        Returns:
            Tuple of (action, amount) where:
                action: 'fold', 'call', 'raise', 'check'
                amount: Amount to bet (0 for fold/check, call amount for call, raise amount for raise)
        """
        if self.is_folded or self.is_all_in:
            return ('check', 0)
        
        # Create a minimal player info list for the interface
        my_public = PokerPlayerPublic(
            id=self.id,
            money=self.money,
            folded=self.is_folded,
            all_in=self.is_all_in,
            bet=self.current_bet,
        )
        
        # Call the new interface method
        poker_action = self.get_action(
            hole_cards=self.get_hole_cards(),
            board=board,
            pot=pot_size,
            current_bet=self.current_bet + current_bet_to_call,
            min_raise=min_raise,
            players=[my_public],  # Only self visible
            my_idx=0,
        )
        
        # Convert PokerAction to (action_str, amount) format
        if poker_action.action_type == ActionType.FOLD:
            return ('fold', 0)
        elif poker_action.action_type == ActionType.CHECK:
            return ('check', 0)
        elif poker_action.action_type == ActionType.CALL:
            return ('call', current_bet_to_call)
        elif poker_action.action_type == ActionType.RAISE:
            # For PokerAgent interface, raise amount is the amount ON TOP of call
            # The poker_action.amount already represents this
            return ('raise', poker_action.amount)
        else:
            return ('check', 0)
    
    def __str__(self):
        """String representation shows human player clearly."""
        return f"{self.id} 👤"

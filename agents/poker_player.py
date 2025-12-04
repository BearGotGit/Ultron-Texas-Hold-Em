"""
PokerPlayer interface for Texas Hold'em agents.
Defines the abstract base class and related types for RL training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
from enum import Enum
import treys


# ============================================================
# Action Types
# ============================================================

class ActionType(Enum):
    """Poker action types."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"  # bet/raise unified


@dataclass
class PokerAction:
    """
    Represents a poker action.
    
    Attributes:
        action_type: Type of action (fold, check, call, raise)
        amount: Bet amount in chips (0 for fold/check, call amount for call, raise amount for raise)
    """
    action_type: ActionType
    amount: int = 0
    
    @classmethod
    def fold(cls) -> "PokerAction":
        return cls(ActionType.FOLD, 0)
    
    @classmethod
    def check(cls) -> "PokerAction":
        return cls(ActionType.CHECK, 0)
    
    @classmethod
    def call(cls, amount: int) -> "PokerAction":
        return cls(ActionType.CALL, amount)
    
    @classmethod
    def raise_to(cls, amount: int) -> "PokerAction":
        return cls(ActionType.RAISE, amount)


# ============================================================
# Player Public Info (what opponents can see)
# ============================================================

@dataclass
class PokerPlayerPublic:
    """
    Public information visible to all players.
    This is what other agents can observe about a player.
    """
    id: str
    money: int
    folded: bool
    all_in: bool
    bet: int  # Current bet in this betting round
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "money": self.money,
            "folded": self.folded,
            "all_in": self.all_in,
            "bet": self.bet,
        }


# ============================================================
# Game State Types
# ============================================================

RoundStage = Literal["pre-flop", "flop", "turn", "river", "showdown"]


@dataclass
class PokerGameState:
    """
    Complete game state at any point.
    """
    players: List["PokerPlayerPublic"]
    pot: int
    round: RoundStage
    board: List[int]  # Treys card integers
    current_bet: int  # Current bet to match
    min_raise: int  # Minimum raise amount
    dealer_position: int
    active_player_idx: int  # Whose turn it is
    game_over: bool = False
    invalid: bool = False  # Truncated flag
    
    def to_dict(self) -> dict:
        return {
            "players": [p.to_dict() for p in self.players],
            "pot": self.pot,
            "round": self.round,
            "board": self.board,
            "current_bet": self.current_bet,
            "min_raise": self.min_raise,
            "dealer_position": self.dealer_position,
            "active_player_idx": self.active_player_idx,
            "game_over": self.game_over,
            "invalid": self.invalid,
        }


# ============================================================
# Pot (for money transfers)
# ============================================================

@dataclass
class Pot:
    """Represents the pot for money transfers."""
    money: int = 0
    
    def add(self, amount: int):
        self.money += amount


# ============================================================
# PokerPlayer Abstract Base Class
# ============================================================

class PokerPlayer(ABC):
    """
    Abstract base class for poker players.
    
    Reusable for different agent types:
    - Monte Carlo agents
    - RL agents (neural network)
    - Human players
    """
    
    def __init__(self, player_id: str, starting_money: int = 1000):
        """
        Initialize a poker player.
        
        Args:
            player_id: Unique identifier for this player
            starting_money: Starting chip stack
        """
        # Public state
        self.id: str = player_id
        self.money: int = starting_money
        self.folded: bool = False
        self.all_in: bool = False
        self.bet: int = 0  # Current bet in this betting round
        self.total_invested: int = 0  # Total invested this hand
        
        # Private state
        self._private_cards: List[int] = []  # Treys card integers
    
    def get_public_info(self) -> PokerPlayerPublic:
        """Get public information visible to all players."""
        return PokerPlayerPublic(
            id=self.id,
            money=self.money,
            folded=self.folded,
            all_in=self.all_in,
            bet=self.bet,
        )
    
    def reset(self):
        """Reset for a new hand (keeps money and id)."""
        self.folded = False
        self.all_in = False
        self.bet = 0
        self.total_invested = 0
        self._private_cards = []
    
    def reset_money(self, starting_money: int):
        """Reset money to starting stack for episodic resets."""
        self.money = starting_money
    
    def reset_bet(self):
        """Reset bet for a new betting round."""
        self.bet = 0
    
    def deal_hand(self, dealt_cards: Tuple[int, int]):
        """
        Receive hole cards.
        
        Args:
            dealt_cards: Tuple of 2 Treys card integers
        """
        self._private_cards = list(dealt_cards)
    
    def get_hole_cards(self) -> List[int]:
        """Get private hole cards."""
        return self._private_cards
    
    def move_money(self, pot: Pot, amount: int) -> int:
        """
        Move up to `amount` from this player to pot.
        Handles all-in partial payments.
        
        Args:
            pot: Destination pot
            amount: Amount to transfer
            
        Returns:
            Actual amount transferred
        """
        # Amount player can actually pay
        pay = min(self.money, amount)
        
        # Deduct from self
        self.money -= pay
        
        # Add to pot
        pot.add(pay)
        
        # Track total chips committed this betting round
        self.bet += pay
        self.total_invested += pay
        
        # All-in state detection
        if self.money == 0:
            self.all_in = True
        
        return pay
    
    def add_winnings(self, amount: int):
        """Add winnings to chip stack."""
        self.money += amount
    
    def fold(self):
        """Fold this hand."""
        self.folded = True
    
    def is_active(self) -> bool:
        """Check if player can still act (not folded, not all-in, has chips)."""
        return not self.folded and not self.all_in and self.money > 0
    
    @abstractmethod
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        """
        Decide on an action given the game state.
        
        Args:
            hole_cards: This player's hole cards (Treys integers)
            board: Community cards (Treys integers)
            pot: Current pot size
            current_bet: Current bet to match
            min_raise: Minimum raise amount
            players: Public info for all players
            my_idx: This player's index in players list
            
        Returns:
            PokerAction describing the chosen action
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id}, ${self.money})"
    
    def __str__(self) -> str:
        return self.id if self.id else self.__class__.__name__


# ============================================================
# SimplePokerAgent - PokerPlayer with legacy PokerAgent API
# ============================================================

class SimplePokerAgent(PokerPlayer):
    """
    A concrete PokerPlayer implementation with backward-compatible
    PokerAgent-style API.
    
    This class provides the same interface as the deprecated PokerAgent
    but inherits from PokerPlayer. It can be used as a drop-in replacement
    for PokerAgent in simulations and tests.
    
    Key differences from PokerPlayer base class:
    - Uses `name` instead of `id` for the player name
    - Uses `chips` instead of `money` for the chip stack
    - Uses `is_folded`/`is_all_in` instead of `folded`/`all_in`
    - Uses `current_bet` instead of `bet` for current round bet
    - Provides `make_decision()` instead of abstract `get_action()`
    - Provides `receive_cards()` instead of `deal_hand()`
    - Provides `place_bet()` instead of `move_money()`
    - Provides `evaluate_hand()` for hand evaluation
    """
    
    def __init__(self, name: Optional[str] = None, starting_chips: int = 1000):
        """
        Initialize a poker agent.
        
        Args:
            name: Optional name for the agent (default: None)
            starting_chips: Starting chip stack (default: 1000)
        """
        super().__init__(player_id=name or "", starting_money=starting_chips)
        self._name = name
        self._evaluator = treys.Evaluator()
    
    # ---- Property aliases for backward compatibility ----
    
    @property
    def name(self) -> Optional[str]:
        """Get the agent's name."""
        return self._name
    
    @name.setter
    def name(self, value: Optional[str]):
        """Set the agent's name."""
        self._name = value
        self.id = value or ""
    
    @property
    def chips(self) -> int:
        """Get current chip stack (alias for money)."""
        return self.money
    
    @chips.setter
    def chips(self, value: int):
        """Set current chip stack (alias for money)."""
        self.money = value
    
    @property
    def hole_cards(self) -> List[int]:
        """Get hole cards (alias for _private_cards)."""
        return self._private_cards
    
    @hole_cards.setter
    def hole_cards(self, value: List[int]):
        """Set hole cards (alias for _private_cards)."""
        self._private_cards = value
    
    @property
    def current_bet(self) -> int:
        """Get current bet this round (alias for bet)."""
        return self.bet
    
    @current_bet.setter
    def current_bet(self, value: int):
        """Set current bet this round (alias for bet)."""
        self.bet = value
    
    @property
    def is_folded(self) -> bool:
        """Check if player has folded (alias for folded)."""
        return self.folded
    
    @is_folded.setter
    def is_folded(self, value: bool):
        """Set folded state (alias for folded)."""
        self.folded = value
    
    @property
    def is_all_in(self) -> bool:
        """Check if player is all-in (alias for all_in)."""
        return self.all_in
    
    @is_all_in.setter
    def is_all_in(self, value: bool):
        """Set all-in state (alias for all_in)."""
        self.all_in = value
    
    # ---- Method aliases for backward compatibility ----
    
    def receive_cards(self, cards: List[int]):
        """
        Receive hole cards.
        
        Args:
            cards: List of 2 card integers
        """
        self._private_cards = cards
    
    def reset_for_new_hand(self):
        """Reset agent state for a new hand."""
        self.reset()
    
    def reset_current_bet(self):
        """Reset current bet for a new betting round."""
        self.reset_bet()
    
    def get_chips(self) -> int:
        """Get current chip stack."""
        return self.money
    
    def add_chips(self, amount: int):
        """
        Add chips to stack (when winning).
        
        Args:
            amount: Number of chips to add
        """
        self.add_winnings(amount)
    
    def place_bet(self, amount: int) -> int:
        """
        Place a bet (deducts chips and tracks bet).
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        # Prevent negative bets
        if amount < 0:
            return 0
        
        actual_bet = min(amount, self.money)
        self.money -= actual_bet
        self.bet += actual_bet
        self.total_invested += actual_bet
        
        if self.money == 0:
            self.all_in = True
        
        return actual_bet
    
    def evaluate_hand(self, board: List[int]) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[float]]:
        """
        Evaluate the strength of the current hand given the board.
        
        Args:
            board: List of community cards
            
        Returns:
            Tuple of (score, hand_class, hand_name, percentage)
        """
        if len(board) < 3:
            return None, None, None, None
        
        score = self._evaluator.evaluate(board, self._private_cards)
        hand_class = self._evaluator.get_rank_class(score)
        hand_name = self._evaluator.class_to_string(hand_class)
        percentage = self._evaluator.get_five_card_rank_percentage(score)
        
        return score, hand_class, hand_name, percentage
    
    def calculate_equity(self, board: List[int], opponent_hands: List[List[int]], 
                         remaining_deck_cards: List[int], num_simulations: int = 1000) -> float:
        """
        Calculate equity (win probability) against opponents using Monte Carlo simulation.
        
        Args:
            board: Current community cards
            opponent_hands: List of opponent hole cards
            remaining_deck_cards: Cards still in the deck
            num_simulations: Number of simulations to run
            
        Returns:
            Float representing win probability (0.0 to 1.0)
        """
        from utils.poker_utils import calculate_hand_equity
        return calculate_hand_equity(self._private_cards, board, opponent_hands, 
                                      remaining_deck_cards, num_simulations)
    
    def _calculate_all_equities(self, board: List[int], hands: List[List[int]], 
                                remaining_deck_cards: List[int], num_simulations: int = 1000) -> List[float]:
        """
        Calculate equity for all hands in the game.
        
        Uses the standalone utility function from utils.poker_utils.
        
        Args:
            board: Current community cards
            hands: List of all player hands
            remaining_deck_cards: Cards still in the deck
            num_simulations: Number of simulations to run
            
        Returns:
            List of win probabilities for each hand
        """
        from utils.poker_utils import calculate_all_equities
        return calculate_all_equities(board, hands, remaining_deck_cards, num_simulations)
    
    def make_decision(self, board: List[int], pot_size: int, current_bet_to_call: int, 
                      min_raise: int) -> Tuple[str, int]:
        """
        Make a betting decision. Override this method to implement custom strategies.
        
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
        # Default strategy: simple equity-based decisions
        if self.folded or self.all_in:
            return ('check', 0)
        
        # If no board cards yet, use simple pre-flop strategy
        if len(board) == 0:
            return self._preflop_decision(current_bet_to_call, min_raise)
        
        # Calculate hand strength
        score, _, hand_name, percentage = self.evaluate_hand(board)
        
        # Simple strategy based on hand strength (percentage is 0=best, 1=worst)
        if percentage < 0.3:  # Strong hand
            if current_bet_to_call == 0:
                return ('raise', min_raise)
            elif current_bet_to_call <= pot_size * 0.5:
                return ('raise', current_bet_to_call + min_raise)
            else:
                return ('call', current_bet_to_call)
        elif percentage < 0.6:  # Medium hand
            if current_bet_to_call == 0:
                return ('check', 0)
            elif current_bet_to_call <= pot_size * 0.3:
                return ('call', current_bet_to_call)
            else:
                return ('fold', 0)
        else:  # Weak hand
            if current_bet_to_call == 0:
                return ('check', 0)
            else:
                return ('fold', 0)
    
    def _preflop_decision(self, current_bet_to_call: int, min_raise: int) -> Tuple[str, int]:
        """Simple pre-flop decision based on hole cards."""
        if current_bet_to_call == 0:
            return ('check', 0)
        elif current_bet_to_call <= self.money * 0.1:
            return ('call', current_bet_to_call)
        else:
            return ('fold', 0)
    
    # ---- Implementation of abstract get_action method ----
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        """
        Decide on an action given the game state.
        
        This implementation adapts the make_decision interface to the
        PokerPlayer get_action interface.
        """
        my_info = players[my_idx]
        to_call = current_bet - my_info.bet
        
        action_str, amount = self.make_decision(board, pot, to_call, min_raise)
        
        if action_str == 'fold':
            return PokerAction.fold()
        elif action_str == 'check':
            return PokerAction.check()
        elif action_str == 'call':
            return PokerAction.call(min(to_call, self.money))
        elif action_str == 'raise':
            # For raise, amount is the raise amount (on top of call)
            raise_total = to_call + amount
            return PokerAction.raise_to(min(raise_total, self.money))
        else:
            return PokerAction.check()
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return self._name if self._name else "Agent"

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

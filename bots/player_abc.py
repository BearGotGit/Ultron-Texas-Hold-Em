from abc import ABC, abstractmethod
from typing import List, Tuple
from bots.interact import ActType, Obs
from game.game_state import PokerPlayerPublic
from utils.treys_int import TREYS_INT

# Treys cards are all `int` - note that!

class PokerPlayer(ABC):
    
    def __init__(self, player_id: str, starting_money: int = 1000):
        # Public state
        self.public = PokerPlayerPublic(
            player_id,
            starting_money,
            False,
            False,
            0
        )

        # Private state
        self._private_cards: List[TREYS_INT] = [] 
    
    @abstractmethod
    def __call__(
        self,
        obs: Obs
    ) -> Tuple[ActType, int]:
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
    
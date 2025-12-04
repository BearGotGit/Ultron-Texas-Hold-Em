
from typing import List, Tuple
from itertools import combinations
import random

from treys import Card, Evaluator, Deck

from bots.player_abc import (
    PokerPlayer,
    PokerPlayerPublic,
    ActType,
)

class RandomAgent(PokerPlayer):
    """
    Simple random agent for testing.
    Makes random decisions with configurable fold probability.
    """
    
    def __init__(
        self,
        player_id: str,
        starting_money: int = 1000,
        fold_prob: float = 0.3,
        raise_prob: float = 0.2,
    ):
        super().__init__(player_id, starting_money)
        self.fold_prob = fold_prob
        self.raise_prob = raise_prob
    
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
        my_info = players[my_idx]
        to_call = current_bet - my_info.bet
        
        if my_info.folded or my_info.all_in:
            return PokerAction.check()
        
        r = random.random()
        
        if to_call <= 0:
            # No bet to call
            if r < self.raise_prob:
                raise_amount = min(min_raise, my_info.money)
                return PokerAction.raise_to(raise_amount)
            return PokerAction.check()
        
        if r < self.fold_prob:
            return PokerAction.fold()
        elif r < self.fold_prob + self.raise_prob:
            raise_amount = min(to_call + min_raise, my_info.money)
            return PokerAction.raise_to(raise_amount)
        else:
            return PokerAction.call(min(to_call, my_info.money))


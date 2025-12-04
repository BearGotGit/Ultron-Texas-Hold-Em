from typing import List, Literal
from dataclasses import dataclass

from utils.treys_int import TREYS_INT

RoundStage = Literal["pre-flop", "flop", "turn", "river", "showdown"]

@dataclass
class PokerPlayerPublic:
    
    id: str
    money: int
    folded: bool
    all_in: bool
    bet: int  # Just this bet round

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "money": self.money,
            "folded": self.folded,
            "all_in": self.all_in,
            "bet": self.bet,
        }
    

@dataclass
class PokerGameState:

    players: List["PokerPlayerPublic"]

    pot: int
    round: RoundStage
    board: List[TREYS_INT]
    to_call: int
    turn_idx: int
    game_over: bool = False
    invalid: bool = False
    
    def to_dict(self) -> dict:
        return {
            "players": [p.to_dict() for p in self.players],

            "pot": self.pot,
            "round": self.round,
            "board": self.board,
            "min_bet_call": self.to_call,
            "player_turn_idx": self.turn_idx,
            "game_over": self.game_over,
            "invalid": self.invalid,
        }

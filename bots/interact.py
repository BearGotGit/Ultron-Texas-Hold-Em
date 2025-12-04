
from dataclasses import dataclass
from typing import List, Literal, Tuple

from utils.treys_int import TREYS_INT


type ActType = Literal["CHECK", "CALL", "FOLD", "RAISE"]

@dataclass
class Obs:
    public_cards: List[TREYS_INT]
    to_call: int
    pot: int
    players_bets: List[int]

    def to_dict(self):
        return {
            "public_cards": self.public_cards,
            "to_call": self.to_call,
            "pot": self.pot,
            "players_bets": self.players_bets
        }
from pprint import pprint
from typing import Tuple
from bots.interact import ActType, Obs
from bots.player_abc import PokerPlayer
from bots.dina_base_bot import BaseBot
from utils.dina_conv import format_from_dina

class Simple(PokerPlayer):
    def __init__ (self):
        pass

    def __call__(
        self,
        obs: Obs
    ) -> Tuple[ActType, int]:
        return ('FOLD', 0)
    
class DinaBotWrapper(BaseBot):
    def __init__(self, bot: PokerPlayer):
        self.bot = bot

    def decide_action(self, game_view) -> Tuple[ActType, int]:

        print("Game view:")
        pprint(game_view)

        obs = format_from_dina(game_view, self.bot)

        print("Obs:\n")
        pprint(obs.to_dict())

        act = self.bot(obs)
        return act
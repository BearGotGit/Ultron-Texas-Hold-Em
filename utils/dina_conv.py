from bots.interact import Obs
from bots.player_abc import PokerPlayer
from game.dina_interact import DinaObs

def format_from_dina(data: DinaObs, me: PokerPlayer) -> Obs:
    """
    {'state': {'ToCall': 10,
            'board': [],
            'hand': 0,
            'phase': 'WAITING',
            'pot': 0,
            'table': {'cardOpen': [],
                        'cardStack': [],
                        'id': 'table-1',
                        'phase': 'WAITING',
                        'players': [{'action': '',
                                    'cards': [{'rank': '', 'suit': ''},
                                                {'rank': '', 'suit': ''}],
                                    'chips': 1000,
                                    'id': 'bpg'}]},
            'toActIdx': -1},
    'type': 'state'}
    """

    return Obs(
        public_cards = data["state"]["board"],
        pot=data["state"]["pot"],
        to_call=data["state"]["ToCall"],
        players_bets=[ p["chips"] for p in data["state"]["table"]["players"] ]
    )
from pprint import pprint
from bots.player_abc import PokerPlayerPublic

player_pub = PokerPlayerPublic("player1", 1000, False, False, 50)
pprint(player_pub.to_dict())

# action = PokerAction()
# action.action_type = ActionType.RAISE
# action.amount = 100
# pprint(action.__dict__)

# pprint(ActionType.FOLD)

# game_state = PokerGameState([player_pub], 200, "flop", [1, 2, 3], 50, 0)
# pprint(game_state.to_dict())

# obs = Obs([1, 2, 3], 50, 200, [50, 100])
# pprint(obs.__dict__)

# act = Act("RAISE", 100)
# pprint(act.__dict__)

# dina_obs = DinaObs({"pot": 200, "phase": "FLOP"})
# pprint(dina_obs.__dict__)

# dina_act = DinaAct("CALL", 50)
# pprint(dina_act.__dict__)
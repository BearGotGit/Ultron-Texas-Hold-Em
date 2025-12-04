import pytest

from connect.player.player1 import RLAgentAdapter


class FakeAgent:
    def __init__(self):
        self.hole_cards = []
        self.chips = 1000
        self.current_bet = 0
        self.is_folded = False
        self.is_all_in = False

    def make_decision(self, board, pot, to_call, min_raise):
        # return uppercase to test normalization in adapter
        # If there's any board card, raise; otherwise call
        if board:
            return "RAISE", max(min_raise, 50)
        return "CALL", 0


def test_rl_adapter_calls_agent_and_normalizes():
    fake = FakeAgent()
    adapter = RLAgentAdapter(fake)

    # create a simple game_view with no board -> expect call
    game_view = {
        "state": {"board": [], "pot": 100},
        "table": {"players": [{"id": "p1"}, {"id": "me", "chips": 500, "cards": [{'rank':14,'suit':'s'}]}]},
        "self_player": {"id": "me", "chips": 500, "cards": [{'rank':14,'suit':'s'}]},
        "my_seat": 1,
    }

    action, amount = adapter.decide_action(game_view)
    assert action in ("call", "raise", "fold")

    # create a game_view with board -> expect raise (normalized)
    game_view2 = {
        "state": {"board": [{'rank':2,'suit':'h'}], "pot": 200},
        "table": {"players": [{"id": "p1"}, {"id": "me", "chips": 500, "cards": [{'rank':14,'suit':'s'}]}]},
        "self_player": {"id": "me", "chips": 500, "cards": [{'rank':14,'suit':'s'}]},
        "my_seat": 1,
    }

    action2, amount2 = adapter.decide_action(game_view2)
    assert action2 == "raise"
    assert isinstance(amount2, int)

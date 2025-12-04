from typing import Any, Dict, Tuple


class BaseBot:
    """
    Minimal bot interface.

    Implement:
        decide_action(self, game_view) -> (action: str, amount: int)

    game_view is a dict with:
        - "state": raw state from server
        - "table": state["table"]
        - "self_player": table["players"][my_seat]
        - "my_seat": int
    """

    def decide_action(self, game_view: Dict[str, Any]) -> Tuple[str, int]:
        raise NotImplementedError

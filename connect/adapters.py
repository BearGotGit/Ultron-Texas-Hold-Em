"""Shared adapters for converting server payloads to agent-friendly formats.

Utilities:
- to_treys_card(card): convert server card (str like 'As' or dict) to Treys int
- normalize_action(action): normalize outgoing action token to server expected format
- game_view_from_server(msg, player_id): convert raw server state message to game_view
"""
from typing import Any, Dict, List, Optional
from treys import Card


def to_treys_card(c: Any) -> int:
    """Convert a server card representation to a Treys int.

    Accepts:
      - treys-style string, e.g. 'As', 'Td'
      - dict with keys `rank` and `suit` (rank may be int or string)
      - integer (passed through)

    Returns 0 on failure.
    """
    if c is None:
        return 0
    if isinstance(c, int):
        return c
    if isinstance(c, str):
        try:
            return Card.new(c)
        except Exception:
            return 0
    if isinstance(c, dict):
        rank = c.get("rank")
        suit = c.get("suit")
        if rank is None or suit is None:
            return 0
        r = str(rank)
        rank_map = {"10": "T", "11": "J", "12": "Q", "13": "K", "14": "A"}
        rc = rank_map.get(r, None)
        if rc is None:
            rc = r[0].upper()
            if rc == '1':
                rc = 'T'
        s = str(suit)[0].lower()
        if s not in 'shdc':
            s = s.lower()[0]
        try:
            return Card.new(f"{rc}{s}")
        except Exception:
            return 0
    return 0



def game_view_from_server(msg: Dict[str, Any], player_id: str) -> Dict[str, Any]:
    """Convert a raw server state message to the `game_view` dict used by bots.

    This normalizes common field name differences between server implementations.
    """
    state = msg.get("state", msg)
    table = state.get("table", {})
    players = table.get("players") or []

    my_seat = None
    for i, p in enumerate(players):
        if not p:
            continue
        pid = p.get("id") or p.get("name")
        if pid == player_id:
            my_seat = i

    self_player = (players or [None])[my_seat] if my_seat is not None else None

    return {
        "state": state,
        "table": table,
        "self_player": self_player,
        "my_seat": my_seat,
    }

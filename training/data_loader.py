"""PyTorch Dataset for poker JSONL per-decision records with card one-hot features.

This loader expects records produced with `--card-format int` (treys ints).
It converts each card into a rank(13)-one-hot + suit(4)-one-hot vector.
"""
from typing import List
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from treys import Card


ACTION_TO_IDX = {"fold": 0, "call": 1, "raise": 2, "check": 3}


RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']


def card_int_to_rank_suit(card_int):
    """Return (rank_char, suit_char) for a treys int card.

    Falls back to string parsing if needed.
    """
    try:
        s = Card.int_to_str(card_int)  # e.g. 'As', 'Td'
    except Exception:
        try:
            s = Card.int_to_pretty_str(card_int)
            # strip ANSI and brackets
            import re

            s = re.sub(r"\x1b\[[0-9;]*m", "", s).strip().strip('[]')
        except Exception:
            return None, None

    if not s or len(s) < 2:
        return None, None

    rank = s[0]
    suit = s[-1].lower()
    return rank, suit


class PokerStepDataset(Dataset):
    """Loads newline-delimited JSON records and produces numeric tensors with card one-hot features.

    Features (per record):
      - hole cards: 2 * (13 rank + 4 suit) = 34
      - board cards: 5 * (13+4) = 85
      - pot, to_call, min_raise, hero_chips = 4

    Total feature size: 123
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.rows: List[dict] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self.rows.append(rec)
                except Exception:
                    continue

    def __len__(self):
        return len(self.rows)

    def _card_onehot(self, card_int):
        rank_vec = [0.0] * len(RANKS)
        suit_vec = [0.0] * len(SUITS)

        if card_int is None or card_int == 0:
            return rank_vec + suit_vec

        rank, suit = card_int_to_rank_suit(card_int)
        if rank in RANKS:
            rank_vec[RANKS.index(rank)] = 1.0
        if suit in SUITS:
            suit_vec[SUITS.index(suit)] = 1.0

        return rank_vec + suit_vec

    def __getitem__(self, idx):
        rec = self.rows[idx]
        hero = rec.get("hero_id")

        hole = rec.get("hole_cards", [])
        # hole cards are expected as ints
        hole_ints = [int(c) if c not in (None, "", []) else 0 for c in hole]
        hole_ints = (hole_ints + [0, 0])[:2]

        board = rec.get("board", [])
        board_ints = [int(c) if c not in (None, "", []) else 0 for c in board]
        board_ints = (board_ints + [0, 0, 0, 0, 0])[:5]

        features = []

        # hole cards
        for c in hole_ints:
            features.extend(self._card_onehot(c))

        # board
        for c in board_ints:
            features.extend(self._card_onehot(c))

        pot = float(rec.get("pot", 0))
        to_call = float(rec.get("to_call", 0))
        min_raise = float(rec.get("min_raise", 0))

        hero_chips = 0.0
        for p in rec.get("players", []):
            if p.get("id") == hero:
                hero_chips = float(p.get("chips", 0))
                break

        features.extend([pot, to_call, min_raise, hero_chips])

        x = torch.tensor(features, dtype=torch.float32)

        action = rec.get("chosen_action", {}).get("action", "fold")
        y = ACTION_TO_IDX.get(action, 0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

"""Helpers to featurize game state for model agents and save/load models."""
from typing import List
import torch
from treys import Card

RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS = ['s','h','d','c']


def card_int_to_rank_suit(card_int):
    try:
        s = Card.int_to_str(card_int)
    except Exception:
        try:
            s = Card.int_to_pretty_str(card_int)
            import re
            s = re.sub(r"\x1b\[[0-9;]*m", "", s).strip().strip('[]')
        except Exception:
            return None, None
    if not s or len(s) < 2:
        return None, None
    rank = s[0]
    suit = s[-1].lower()
    return rank, suit


def card_onehot_vec(card_int):
    rank_vec = [0.0]*len(RANKS)
    suit_vec = [0.0]*len(SUITS)
    if card_int in (None, 0, "", []):
        return rank_vec + suit_vec
    r,s = card_int_to_rank_suit(card_int)
    if r in RANKS:
        rank_vec[RANKS.index(r)] = 1.0
    if s in SUITS:
        suit_vec[SUITS.index(s)] = 1.0
    return rank_vec + suit_vec


def featurize_state(agent, agents: List, board: List[int], pot: int, to_call: int, min_raise: int):
    """Return torch.FloatTensor feature vector matching `training.data_loader.PokerStepDataset`.

    Order: hole(2)*(13+4) + board(5)*(13+4) + [pot,to_call,min_raise,hero_chips]
    """
    hero = agent.name
    # hole
    hole = agent.get_hole_cards() or []
    hole_ints = [int(h) if h not in (None, '', []) else 0 for h in hole]
    hole_ints = (hole_ints + [0,0])[:2]

    board_ints = [int(c) if c not in (None,'',[]) else 0 for c in (board or [])]
    board_ints = (board_ints + [0,0,0,0,0])[:5]

    feats = []
    for c in hole_ints:
        feats.extend(card_onehot_vec(c))
    for c in board_ints:
        feats.extend(card_onehot_vec(c))

    hero_chips = 0.0
    for p in agents:
        if p.name == hero:
            hero_chips = float(p.get_chips())
            break

    feats.extend([float(pot), float(to_call), float(min_raise), float(hero_chips)])

    return torch.tensor(feats, dtype=torch.float32)

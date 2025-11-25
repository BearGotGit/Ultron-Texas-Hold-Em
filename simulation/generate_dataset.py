"""
Generate training data by running headless AI vs AI hands and recording decision steps.

Outputs newline-delimited JSON records (one per decision step).
"""
import argparse
import json
import os
import time
import uuid
import random
import re
from treys import Card

from agents import PokerAgent
from simulation.poker_simulator import TexasHoldemSimulation


def card_to_str(c, fmt="compact"):
    """Return a string for card `c`.

    fmt: 'compact' -> 'As', 'Td' etc; 'pretty' -> treys pretty string; 'int' -> integer code
    """
    if c is None:
        return ""

    if fmt == "pretty":
        try:
            return Card.int_to_pretty_str(c)
        except Exception:
            return str(c)

    # compact
    if fmt == "int":
        # return raw integer code
        try:
            return int(c)
        except Exception:
            try:
                return int(str(c))
            except Exception:
                return c

    if hasattr(Card, "int_to_str"):
        try:
            return Card.int_to_str(c)
        except Exception:
            pass

    # Fall back to stripping ANSI from pretty string
    try:
        s = Card.int_to_pretty_str(c)
        # remove ANSI escape sequences
        s = re.sub(r"\x1b\[[0-9;]*m", "", s)
        # remove surrounding brackets if present
        s = s.strip()
        s = s.strip("[]")
        return s
    except Exception:
        return str(c)


def make_record(hand_id, players_state, hero_id, hole_cards, board, pot, to_call, min_raise, legal_actions, chosen_action, equity=None, card_fmt="compact"):
    return {
        "meta": {"hand_id": hand_id, "ts": time.time()},
        "players": players_state,
        "hero_id": hero_id,
        "hole_cards": [card_to_str(c, fmt=card_fmt) for c in (hole_cards or [])],
        "board": [card_to_str(c, fmt=card_fmt) for c in (board or [])],
        "pot": pot,
        "to_call": to_call,
        "min_raise": min_raise,
        "legal_actions": legal_actions,
        "chosen_action": chosen_action,
        "equity": equity,
    }


def default_legal_actions(agent, to_call, min_raise):
    actions = []
    if to_call == 0:
        actions.append("check")
        actions.append("raise")
    else:
        actions.append("fold")
        if to_call <= agent.get_chips():
            actions.append("call")
        if agent.get_chips() >= to_call + min_raise:
            actions.append("raise")
    return actions


def players_state_snapshot(agents):
    state = []
    for i, a in enumerate(agents):
        state.append({
            "id": a.name or f"player_{i}",
            "chips": a.get_chips(),
            "current_bet": a.current_bet,
            "total_invested": a.total_invested,
            "is_folded": a.is_folded,
            "is_all_in": a.is_all_in,
        })
    return state


def generate(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if args.out and os.path.dirname(args.out) else None

    agents = [PokerAgent(name=f"Player{i+1}", starting_chips=args.starting_chips) for i in range(args.players)]
    game = TexasHoldemSimulation(agents, small_blind=args.small_blind, big_blind=args.big_blind)

    with open(args.out, "w", encoding="utf-8") as fout:
        for hand_idx in range(args.hands):
            hand_id = str(uuid.uuid4())
            # reset and deal
            game.reset_for_new_hand()
            game.dealer_position = (game.dealer_position + 1) % len(agents)
            sb_pos, bb_pos = game.post_blinds()
            game.deal_hole_cards()

            # Wrap agents' make_decision to record each decision
            originals = {}

            def make_wrapper(agent_index):
                agent = agents[agent_index]

                originals[agent_index] = agent.make_decision

                def wrapper(board, pot_size, current_bet_to_call, min_raise):
                    legal = default_legal_actions(agent, current_bet_to_call, min_raise)

                    # Optionally compute equity for this agent (expensive)
                    equity = None
                    if args.include_equity:
                        try:
                            equity = agent.calculate_equity(board, [h for i,h in enumerate([p.get_hole_cards() for p in agents]) if i!=agent_index], game.deck.cards[:], num_simulations=args.equity_sims)
                        except Exception:
                            equity = None

                    action, amount = originals[agent_index](board, pot_size, current_bet_to_call, min_raise)

                    rec = make_record(
                        hand_id=hand_id,
                        players_state=players_state_snapshot(agents),
                        hero_id=agent.name,
                        hole_cards=agent.get_hole_cards(),
                        board=board,
                        pot=pot_size,
                        to_call=current_bet_to_call,
                        min_raise=min_raise,
                        legal_actions=legal,
                        chosen_action={"action": action, "amount": amount},
                        equity=equity,
                        card_fmt=args.card_format,
                    )
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()

                    return action, amount

                return wrapper

            # Install wrappers
            for idx in range(len(agents)):
                agents[idx].make_decision = make_wrapper(idx)

            # Run hand flow (preflop -> flop -> turn -> river)
            game.run_betting_round("Pre-flop")

            if len([a for a in agents if not a.is_folded]) > 1:
                game.deal_flop()
                game.run_betting_round("Flop")

            if len([a for a in agents if not a.is_folded]) > 1:
                game.deal_turn()
                game.run_betting_round("Turn")

            if len([a for a in agents if not a.is_folded]) > 1:
                game.deal_river()
                game.run_betting_round("River")

            # Award pot to update chips
            game.award_pot()

            # Restore originals (in case)
            for idx, orig in originals.items():
                agents[idx].make_decision = orig

            if (hand_idx + 1) % 50 == 0:
                print(f"Generated {hand_idx+1} / {args.hands} hands")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=1000, help="Number of hands to simulate")
    p.add_argument("--players", type=int, default=4, help="Number of players per hand")
    p.add_argument("--starting-chips", type=int, default=1000)
    p.add_argument("--out", type=str, default="data/processed/dataset.jsonl")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--card-format", type=str, choices=["compact", "pretty", "int"], default="compact", help="Card output format: compact (As) or pretty (ANSI colored) or int (treys int code)")
    p.add_argument("--include-equity", action="store_true", help="Compute Monte Carlo equity at each decision (slow)")
    p.add_argument("--equity-sims", type=int, default=200, help="Monte Carlo sims for per-decision equity when enabled")
    p.add_argument("--small-blind", type=int, default=5)
    p.add_argument("--big-blind", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    print(f"Generating {args.hands} hands -> {args.out}")
    generate(args)


if __name__ == "__main__":
    main()

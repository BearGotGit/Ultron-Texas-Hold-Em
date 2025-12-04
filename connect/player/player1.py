import argparse
import json
import sys
import threading
import time
# --- Fix for PPOConfig unpickling errors ---
try:
    import training.train_rl_model as _train_module
    import __main__ as _main
    if not hasattr(_main, "PPOConfig"):
        _main.PPOConfig = _train_module.PPOConfig
except Exception:
    pass
# ------------------------------------------
try:
    from websocket import WebSocketApp, WebSocketConnectionClosedException
except Exception:
    WebSocketApp = None
    WebSocketConnectionClosedException = Exception
import ssl
from dotenv import load_dotenv
import os
import random
from typing import Optional, Tuple, Any, Dict, List
from pathlib import Path

from connect.adapters import to_treys_card, normalize_action, game_view_from_server

load_dotenv()


def safe_card_str(c: dict) -> str:
    rank = c.get("rank", "") or ""
    suit = c.get("suit", "") or ""
    suit_char = suit[0] if isinstance(suit, str) and len(suit) > 0 else ""
    return f"{rank}{suit_char}"


# =========================
#  Bot interface + example
# =========================

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


class RandomBot(BaseBot):
    """
    Example bot:
      - Sometimes folds
      - Sometimes calls
      - Sometimes raises a small random amount
    """

    def __init__(self, max_raise: int = 20):
        self.max_raise = max_raise

    def decide_action(self, game_view: Dict[str, Any]) -> Tuple[str, int]:
        # VERY dumb strategy, just to show the interface.
        # You can inspect game_view["state"] here for something smarter.
        choice = random.random()
        if choice < 0.15:
            return "FOLD", 0
        elif choice < 0.85:
            return "CALL", 0
        else:
            amt = random.randint(1, self.max_raise)
            return "RAISE", amt


class RLAgentAdapter(BaseBot):
    """Adapter to wrap an `agents.rl_agent.RLAgent` instance and expose
    the older `decide_action(game_view)` interface used by this client.
    """

    def __init__(self, rl_agent):
        self.rl_agent = rl_agent

    def decide_action(self, game_view: Dict[str, Any]) -> Tuple[str, int]:
        state = game_view.get("state", {})
        table = game_view.get("table", {})
        self_player = game_view.get("self_player") or {}

        # Convert hole cards and board to treys ints
        hole_cards_raw = self_player.get("cards") or self_player.get("hole_cards") or []
        try:
            treys_hole = [to_treys_card(c) for c in hole_cards_raw]
        except Exception:
            treys_hole = []

        board_raw = state.get("board", [])
        try:
            treys_board = [to_treys_card(c) for c in board_raw]
        except Exception:
            treys_board = []

        agent = self.rl_agent
        # populate minimal state the RLAgent expects
        try:
            agent.hole_cards = treys_hole
            agent.chips = int(self_player.get("chips", getattr(agent, "chips", 1000)))
            agent.current_bet = int(self_player.get("bet", self_player.get("current_bet", getattr(agent, "current_bet", 0))))
            agent.is_folded = bool(self_player.get("folded") or self_player.get("isFolded") or self_player.get("fold", False))
            agent.is_all_in = bool(self_player.get("allIn") or self_player.get("isAllIn") or self_player.get("all_in", False))
        except Exception:
            pass

        # Compute to-call
        players = table.get("players") or []
        bets = [p.get("bet", 0) for p in players if p]
        current_bet = max(bets) if bets else 0
        my_bet = int(self_player.get("bet", 0)) if self_player else 0
        current_bet_to_call = max(0, current_bet - my_bet)

        min_raise = state.get("minRaise") or state.get("min_raise") or table.get("minRaise") or 0

        try:
            action, amount = agent.make_decision(treys_board, state.get("pot", 0), current_bet_to_call, min_raise)
        except Exception:
            return "CALL", 0

        act = (action or "").strip()
        act_norm = normalize_action(act)
        if act_norm != "raise":
            amount = 0
        return act_norm, int(amount)


class PlayerClient:
    def __init__(
        self,
        player_id: str,
        bot: Optional[BaseBot] = None,
        baseurl: Optional[str] = None,
        api_key: Optional[str] = None,
        table_id: Optional[str] = None,
        insecure_ssl: bool = True,
    ):
        self.player_id = player_id
        self.bot = bot

        # Connection parameters (CLI overrides env)
        self.baseurl = baseurl or os.getenv("BASEURL")
        self.api_key = api_key or os.getenv("APIKEY")
        self.table_id = table_id or os.getenv("TABLEID")
        self.insecure_ssl = insecure_ssl

        self.ws: Optional[Any] = None
        self.my_seat: Optional[int] = None
        self.to_act_idx: int = -1
        self.phase: str = "WAITING"
        self.latest_state: Optional[dict] = None

        self.lock = threading.Lock()

    # ---- websocket callbacks ----
    def on_open(self, ws):
        print(f"[{self.player_id}] Connected")
        ws.send(json.dumps({"type": "join"}))

    def on_error(self, ws, error):
        print(f"[{self.player_id}] error:", error)

    def on_close(self, ws, status_code, msg):
        print(f"[{self.player_id}] connection closed:", status_code, msg)

    def on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"[{self.player_id}] non-JSON:", message)
            return

        if data.get("type") != "state":
            print(f"[{self.player_id}] msg:", data)
            return

        state = data["state"]
        table = state["table"]
        pot = state["pot"]
        phase = state["phase"]
        board = state["board"]
        to_act_idx = state["toActIdx"]
        hand_no = state.get("hand", 0)

        players = table.get("players") or []
        pretty_players = []
        my_seat = None
        for i, p in enumerate(players):
            if p is None:
                continue
            pid = p.get("id")
            chips = p.get("chips")
            cards = p.get("cards") or []
            cards_str = ""
            if cards:
                cards_str = " " + " ".join(safe_card_str(c) for c in cards)
            if pid == self.player_id:
                my_seat = i
            pretty_players.append(f"   seat {i}: {pid}:{chips}{cards_str}")

        with self.lock:
            self.my_seat = my_seat
            self.to_act_idx = to_act_idx
            self.phase = phase
            self.latest_state = state

        print(f"\n=== STATE for {self.player_id} (hand #{hand_no}) ===")
        print(f"Phase: {phase}  Pot: {pot}")
        if board:
            print("Board:", "[" + " ".join(safe_card_str(c) for c in board) + "]")
        else:
            print("Board: []")
        print("Players:")
        for line in pretty_players:
            print(line)
        print(f"ToActIdx: {to_act_idx}   (your seat: {my_seat})")

        # show commands always, to make debugging easier
        print(f"[{self.player_id}] Commands when it's your turn:")
        print("  c         -> check/call")
        print("  f         -> fold")
        print("  r <amt>   -> raise by <amt> (raise size, not total bet)")
        print("  q         -> quit client")

        # hint when it's really your turn
        if my_seat is not None and to_act_idx == my_seat and phase not in ("WAITING", "SHOWDOWN"):
            print(f"[{self.player_id}] 💡 It's your turn!")

        # If a bot is attached, let it act automatically when it's our turn
        if (
            self.bot is not None
            and my_seat is not None
            and to_act_idx == my_seat
            and phase not in ("WAITING", "SHOWDOWN")
        ):
            self._maybe_run_bot_action(state, table, my_seat)

    def _maybe_run_bot_action(self, state: dict, table: dict, my_seat: int):
        """Call the bot's decide_action hook and send the action."""
        game_view = {
            "state": state,
            "table": table,
            "self_player": (table.get("players") or [None])[my_seat],
            "my_seat": my_seat,
        }

        try:
            action, amount = self.bot.decide_action(game_view)
        except Exception as e:
            print(f"[{self.player_id}] bot error:", e)
            return

        action = (action or "").upper().strip()
        if action not in ("CALL", "FOLD", "RAISE"):
            print(f"[{self.player_id}] bot returned invalid action: {action}")
            return

        if action != "RAISE":
            amount = 0

        print(f"[{self.player_id}] 🤖 Bot acting: {action} {amount}")
        self.send_action(action, amount)

    # ---- send action ----
    def send_action(self, action: str, amount: int = 0):
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            print(f"[{self.player_id}] cannot send, not connected")
            return
        msg = {
            "type": "act",
            "action": normalize_action(action),
            "amount": amount,
        }
        try:
            self.ws.send(json.dumps(msg))
        except WebSocketConnectionClosedException as e:
            print(f"[{self.player_id}] send failed:", e)

    # ---- threads ----
    def run_ws(self):
        # Build URL from provided values (CLI/env). Require baseurl/api_key/table.
        if not self.baseurl or not self.api_key or not self.table_id:
            print(f"[{self.player_id}] Missing connection parameters: BASEURL/APIKEY/TABLEID")
            print("Set them via environment variables or CLI flags.")
            return

        url = f"{self.baseurl.rstrip('/')}" + f"/ws?apiKey={self.api_key}&table={self.table_id}&player={self.player_id}"
        print(f"[{self.player_id}] Connecting to {url}")

        self.ws = WebSocketApp(
            url,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )

        # SSL options: allow disabling verification for dev, but prefer secure by default
        sslopt = None
        if self.insecure_ssl:
            sslopt = {"cert_reqs": ssl.CERT_NONE}

        if sslopt is not None:
            self.ws.run_forever(sslopt=sslopt)
        else:
            self.ws.run_forever()

    def run_input_loop(self):
        """
        Human CLI loop.
        If a bot is attached, this method is not used.
        """
        if self.bot is not None:
            print(f"[{self.player_id}] Bot mode enabled – no human input loop.")
            return

        # simple input loop; you can type even if it's not your turn, we gate it
        while True:
            try:
                cmd = input(f"[{self.player_id}] your move (c/f/r <amt>/q): ").strip()
            except EOFError:
                break

            if cmd == "":
                continue
            if cmd.lower() == "q":
                print(f"[{self.player_id}] quitting client…")
                if self.ws:
                    self.ws.close()
                break

            with self.lock:
                my_seat = self.my_seat
                to_act_idx = self.to_act_idx
                phase = self.phase

            if phase in ("WAITING", "SHOWDOWN") or my_seat is None or to_act_idx != my_seat:
                print(f"[{self.player_id}] Not your turn right now.")
                continue

            parts = cmd.split()
            code = parts[0].lower()
            amt = 0
            if len(parts) >= 2:
                try:
                    amt = int(parts[1])
                except ValueError:
                    print(f"[{self.player_id}] invalid amount, using 0")
                    amt = 0

            if code == "c":
                self.send_action("CALL", 0)
            elif code == "f":
                self.send_action("FOLD", 0)
            elif code == "r":
                self.send_action("RAISE", amt)
            else:
                print(f"[{self.player_id}] unknown command, use c/f/r or q")

        print(f"[{self.player_id}] client closed")


def build_bot_from_args(args: List[str], model_checkpoint_env: Optional[str] = None) -> Optional[BaseBot]:
    """
    Build a bot from CLI args. Supports:
      - randbot / random
      - rl [checkpoint]
    """
    if not args:
        return None

    bot_name = args[0].lower()
    if bot_name in ("randbot", "random"):
        return RandomBot()

    # RL bot: `rl [checkpoint_path]`
    if bot_name in ("rl", "rlbot"):
        try:
            from agents.rl_agent import RLAgent
        except Exception:
            print("[main] RL agent support not available (agents.rl_agent import failed).")
            return None

        cli_ckpt = args[1] if len(args) >= 2 else None

        def find_latest_checkpoint(candidate: Optional[str]) -> Optional[str]:
            if candidate:
                p = Path(candidate)
                if p.is_file():
                    return str(p)
                if p.is_dir():
                    search_dir = p
                else:
                    search_dir = Path("checkpoints")
            else:
                env_ckpt = model_checkpoint_env or os.getenv("MODEL_CHECKPOINT")
                if env_ckpt:
                    p = Path(env_ckpt)
                    if p.is_file():
                        return str(p)
                    if p.is_dir():
                        search_dir = p
                    else:
                        search_dir = Path("checkpoints")
                else:
                    search_dir = Path("checkpoints")

            if not search_dir.exists() or not search_dir.is_dir():
                return None

            patterns = ["*.pt", "*.pth"]
            candidates = []
            for pat in patterns:
                candidates.extend(list(search_dir.glob(pat)))

            if not candidates:
                return None

            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])

        chosen = find_latest_checkpoint(cli_ckpt)
        if chosen is None:
            print("[main] No RL checkpoint found in CLI/ENV/checkpoints/. Running in human mode.")
            return None

        print(f"[main] Using RL checkpoint: {chosen}")
        try:
            agent = RLAgent.from_checkpoint(checkpoint_path=str(chosen), player_id="Ultron", starting_money=1000)
            return RLAgentAdapter(agent)
        except Exception as e:
            print(f"[main] Failed to load RL agent: {e}")
            return None

    print(f"[main] Unknown bot '{bot_name}', running in human mode.")
    return None


def main():
    parser = argparse.ArgumentParser(description="WebSocket player client")
    parser.add_argument("player_id", type=str, help="Player ID to join as")
    parser.add_argument("bot", nargs="*", help="Optional bot args (e.g. randbot or 'rl [checkpoint]')")
    parser.add_argument("--baseurl", type=str, default=os.getenv("BASEURL"), help="WebSocket base URL (wss://host)")
    parser.add_argument("--apikey", type=str, default=os.getenv("APIKEY"), help="API key for server")
    parser.add_argument("--table", type=str, default=os.getenv("TABLEID"), help="Table ID to join")
    parser.add_argument("--model-checkpoint", type=str, default=os.getenv("MODEL_CHECKPOINT"), help="Checkpoint path or dir for RL bot")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification (dev-only)")

    args = parser.parse_args()

    player_id = args.player_id
    bot = build_bot_from_args(args.bot or [], model_checkpoint_env=args.model_checkpoint)

    client = PlayerClient(player_id, bot=bot, baseurl=args.baseurl, api_key=args.apikey, table_id=args.table, insecure_ssl=args.insecure_ssl)

    t_ws = threading.Thread(target=client.run_ws, daemon=True)
    t_ws.start()

    # give ws a moment to connect
    time.sleep(1.0)

    if bot is None:
        client.run_input_loop()
    else:
        print(f"[{player_id}] Running in BOT mode. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"[{player_id}] KeyboardInterrupt, closing…")
            if client.ws:
                client.ws.close()


if __name__ == "__main__":
    main()

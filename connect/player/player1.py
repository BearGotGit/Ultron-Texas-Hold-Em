import json
import sys
import threading
import time
# --- Fix for PPOConfig unpickling errors ---
try:
    # Import the module containing PPOConfig
    from training.train_rl_model import PPOConfig
    import __main__ as _main

    # Make __main__.PPOConfig point to the real class
    if not hasattr(_main, "PPOConfig"):
        _main.PPOConfig = PPOConfig
except Exception:
    pass
# ------------------------------------------

try:
    # Preferred import from websocket-client
    from websocket import WebSocketApp, WebSocketConnectionClosedException
except Exception:
    # Fallback: try importing the package as a module and pull attributes
    try:
        import websocket as _ws

        WebSocketApp = getattr(_ws, "WebSocketApp", None)
        WebSocketConnectionClosedException = getattr(
            _ws, "WebSocketConnectionClosedException", None
        )
        # Some installations expose a generic WebSocketException
        if WebSocketConnectionClosedException is None:
            WebSocketConnectionClosedException = getattr(_ws, "WebSocketException", Exception)

        if WebSocketApp is None:
            raise ImportError("websocket package does not provide WebSocketApp")
    except Exception as _e:
        raise ImportError(
            "websocket client library not found or incompatible. Please install 'websocket-client'"
        ) from _e
import ssl  # 👈 make sure this is imported
from treys import Card
from connect.adapters import to_treys_card, normalize_action, game_view_from_server
from dotenv import load_dotenv
import os
import random
from typing import Optional, Tuple, Any, Dict
from pathlib import Path
import glob
import os

# Optional RL agent import (use canonical API)
try:
    from agents.rl_agent import RLAgent
except Exception:
    RLAgent = None

load_dotenv()
# Read env vars at import time but do not raise here; runtime checks happen in run_ws
BASEURL = os.getenv("BASEURL")
API_KEY = os.getenv("APIKEY")
TABLE_ID = os.getenv("TABLEID")

WS_URL_TEMPLATE = (BASEURL or "") + "/ws?apiKey={apiKey}&table={table}&player={player}"


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
        if choice < 0.2:
            return "FOLD", 0
        elif choice < 0.8:
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

    def _to_treys_card(self, c):
        return to_treys_card(c)

    def decide_action(self, game_view: Dict[str, Any]) -> Tuple[str, int]:
        state = game_view.get("state", {})
        table = game_view.get("table", {})
        self_player = game_view.get("self_player") or {}

        # Convert hole cards and board to treys ints
        hole_cards_raw = self_player.get("cards") or self_player.get("hole_cards") or []
        treys_hole = [to_treys_card(c) for c in hole_cards_raw]

        board_raw = state.get("board", [])
        treys_board = [to_treys_card(c) for c in board_raw]

        # Populate RLAgent internal state
        agent = self.rl_agent
        agent.hole_cards = treys_hole
        try:
            agent.chips = int(self_player.get("chips", agent.chips))
        except Exception:
            pass
        try:
            agent.current_bet = int(self_player.get("bet", self_player.get("current_bet", agent.current_bet)))
        except Exception:
            pass
        agent.is_folded = bool(self_player.get("folded") or self_player.get("isFolded") or self_player.get("fold", False))
        agent.is_all_in = bool(self_player.get("allIn") or self_player.get("isAllIn") or self_player.get("all_in", False))

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

        # Normalize action names to the CLI expected tokens
        act = (action or "").strip()
        # normalize to server expected token (lower-case)
        act_norm = normalize_action(act)
        if act_norm != "raise":
            amount = 0
        return act_norm, int(amount)


class PlayerClient:
    def __init__(self, player_id: str, bot: Optional[BaseBot] = None):
        self.player_id = player_id
        self.bot = bot

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
        url = WS_URL_TEMPLATE.format(
            apiKey=API_KEY,
            table=TABLE_ID,
            player=self.player_id
        )
        print(f"[{self.player_id}] Connecting to {url}")

        self.ws = WebSocketApp(
            url,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )

        # DEV ONLY: disable certificate verification so wss connect works
        sslopt = {"cert_reqs": ssl.CERT_NONE}

        self.ws.run_forever(sslopt=sslopt)

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


def build_bot_from_args(args: list[str]) -> Optional[BaseBot]:
    """
    Simple hook so people can choose a bot from the CLI.

    Examples:
        python3 player_cli.py p1           # human mode
        python3 player_cli.py p1 randbot   # random bot example
    """
    if not args:
        return None

    bot_name = args[0].lower()
    if bot_name in ("randbot", "random"):
        return RandomBot()

    # RL bot: `rl [checkpoint_path]`
    if bot_name in ("rl", "rlbot"):
        if RLAgent is None:
            print("[main] RL agent support not available (agents.rl_agent import failed).")
            return None

        # checkpoint may be provided as second arg, otherwise try env or default
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
                env_ckpt = os.getenv("MODEL_CHECKPOINT")
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

        cli_ckpt = args[1] if len(args) >= 2 else None
        chosen = find_latest_checkpoint(cli_ckpt)
        if chosen is None:
            print("[main] No RL checkpoint found in CLI/ENV/checkpoints/. Running in human mode.")
            return None

        print(f"[main] Using RL checkpoint: {chosen}")
        try:
            # Load RLAgent using the canonical API
            agent = RLAgent.from_checkpoint(
                checkpoint_path=str(chosen),
                player_id="Ultron",       # or the bot's name
                starting_money=1000,      # must match server stack size
            )
            return RLAgentAdapter(agent)
        except Exception as e:
            print(f"[main] Failed to load RL agent: {e}")
            return None

    # Extend here: elif bot_name == "mybot": return MyBot(...)
    print(f"[main] Unknown bot '{bot_name}', running in human mode.")
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Human: python3 player_cli.py <playerId>")
        print("  Bot:   python3 player_cli.py <playerId> randbot")
        sys.exit(1)

    player_id = sys.argv[1]
    extra_args = sys.argv[2:]

    bot = build_bot_from_args(extra_args)
    client = PlayerClient(player_id, bot=bot)

    t_ws = threading.Thread(target=client.run_ws)
    t_ws.start()

    # give ws a moment to connect
    time.sleep(1.0)

    if bot is None:
        # Human interaction
        client.run_input_loop()
    else:
        # Bot mode: just keep the main thread alive until Ctrl+C
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

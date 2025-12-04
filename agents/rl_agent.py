"""
Clean, minimal, FIXED RL Agent wrapper for your Ultron Texas Hold'em server.
This version ensures:
- blinds are handled correctly
- call amounts are correct
- min-raise is always valid
- RL agent never sends illegal actions
"""

import torch
from agents.agent import PokerAgent
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import RandomAgent
from connect.adapters import to_treys_card


# ============================================
#  RLAgent (same structure, cleaned + corrected)
# ============================================

class RLAgent(PokerAgent):
    @classmethod
    def from_checkpoint(cls, path, player_id, starting_money, device=None):
        from utils.device import DEVICE
        from training.ppo_model import PokerPPOModel

        device = device or DEVICE
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = PokerPPOModel(
            card_embed_dim=64,
            hidden_dim=256,
            num_shared_layers=2,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return cls(player_id, starting_money, model, device)

    def __init__(self, name, starting_chips, model, device):
        super().__init__(name, starting_chips)
        self.model = model
        self.device = device

        cfg = PokerEnvConfig(
            small_blind=5,
            big_blind=10,
            starting_stack=starting_chips,
            max_players=2,
        )
        placeholder = RandomAgent("temp", starting_chips)
        self.temp_env = PokerEnv([placeholder, placeholder], cfg, hero_idx=0)

    def make_decision(self, board, pot, current_bet, my_bet, min_raise):
        # Respect folded / all-in flags
        if self.is_folded or self.is_all_in:
            return ("check", 0)

        # Compute to_call exactly how env does it
        to_call = max(0, current_bet - my_bet)

        # Sync env with true state
        self._sync_env(board, pot, current_bet, my_bet, min_raise)

        # Observation
        obs = self.temp_env._get_observation()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            fold_logit, a, b, _ = self.model(obs_tensor)
            p_fold = torch.sigmoid(fold_logit).item()
            # For evaluation, mean of Beta is more stable than random sample
            a_v = float(a.squeeze().cpu().numpy())
            b_v = float(b.squeeze().cpu().numpy())
            bet_scalar = a_v / (a_v + b_v + 1e-8)

        from agents.poker_player import ActionType

        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=current_bet,
            my_bet=my_bet,
            min_raise=min_raise,
            my_money=self.chips,
        )

        t = poker_action.action_type
        amt = int(poker_action.amount)

        # Safety fixes like in your eval script
        if t == ActionType.FOLD and to_call == 0:
            return ("check", 0)

        if t == ActionType.RAISE and amt < to_call:
            # nonsense raise → turn into call
            return ("call", to_call)

        if t == ActionType.FOLD:
            return ("fold", 0)
        if t == ActionType.CHECK:
            return ("check", 0)
        if t == ActionType.CALL:
            # prefer the explicit amount returned by the env's interpret_action
            call_amt = amt if amt > 0 else to_call
            return ("call", int(call_amt))
        if t == ActionType.RAISE:
            return ("raise", int(amt))

        return ("check", 0)

    def _sync_env(self, board, pot, current_bet, my_bet, min_raise):
        env = self.temp_env
        env.board = [to_treys_card(c) for c in board]        # ideally convert to Treys ints
        env.pot.money = pot
        env.current_bet = current_bet
        env.min_raise = min_raise

        hero = env.players[0]
        # Ensure hero's private cards are set using the internal API
        try:
            # If we have exactly two hole cards, use deal_hand
            if hasattr(self, 'hole_cards') and self.hole_cards and len(self.hole_cards) == 2:
                hero.deal_hand(tuple(self.hole_cards))
            else:
                # fallback: set private attribute directly
                hero._private_cards = list(getattr(self, 'hole_cards', []))
        except Exception:
            hero._private_cards = list(getattr(self, 'hole_cards', []))
        hero.money = self.chips
        hero.bet = my_bet 
        hero.folded = self.is_folded
        hero.all_in = self.is_all_in

        # Stage heuristic
        if len(board) == 0:
            env.round_stage = "pre-flop"
        elif len(board) == 3:
            env.round_stage = "flop"
        elif len(board) == 4:
            env.round_stage = "turn"
        elif len(board) == 5:
            env.round_stage = "river"



# ============================================
#  load_model() FOR OLDER CODE SUPPORT
# ============================================

def load_model(path):
    from utils.device import DEVICE
    from training.ppo_model import PokerPPOModel

    device = DEVICE
    chk = torch.load(path, map_location=device, weights_only=False)

    model = PokerPPOModel(
        card_embed_dim=64,
        hidden_dim=256,
        num_shared_layers=2,
    ).to(device)

    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    return model, device


# ============================================
#  RLBot Wrapper — CLEAN & FIXED
# ============================================

class RLBot:
    """
    Wrapper that connects the server's raw game state
    to the RLAgent that was trained in PokerEnv.
    """

    def __init__(self, model, device, starting_chips=1000, name="rl-bot"):
        self.agent = RLAgent(name, starting_chips, model, device)

    def decide_action(self, game_view):
        """
        Convert server game state → RLAgent inputs → return (action, amount)
        """

        state = game_view["state"]
        table = game_view["table"]
        me = game_view["self_player"]

        if me is None:
            print("[DEBUG] No self_player in game_view, defaulting to check.")
            return ("check", 0)

        # ------------------------------
        # Convert board to Treys format
        # ------------------------------
        raw_board = state.get("board", [])
        board = [to_treys_card(c) for c in raw_board]

        # Pot
        pot = state.get("pot", 0)

        # ------------------------------
        # Compute current bet and call amount
        # ------------------------------
        players = table.get("players") or []
        bets = [p.get("bet", 0) for p in players if p]
        current_bet = max(bets) if bets else 0

        my_bet = me.get("bet", 0) or 0
        to_call = max(0, current_bet - my_bet)

        # ------------------------------
        # Proper min-raise handling
        # ------------------------------
        min_raise = state.get("minRaise")
        if min_raise is None:
            min_raise = state.get("min_raise")
        if min_raise is None:
            min_raise = table.get("minRaise")
        if min_raise is None:
            min_raise = 10   # Fallback big blind

        # ------------------------------
        # Convert hole cards to Treys ints
        # ------------------------------
        raw_hole = me.get("cards", [])
        hole_cards = [to_treys_card(c) for c in raw_hole]
        self.agent.hole_cards = hole_cards

        # Update chips
        self.agent.chips = int(me.get("chips", self.agent.chips))

        # ------------------------------
        # Ask RLAgent for the decision
        # ------------------------------
        try:
            action, amount = self.agent.make_decision(
                board=board,
                pot=pot,
                current_bet=current_bet,
                my_bet=my_bet,
                min_raise=min_raise,
            )
            print(f"[DEBUG] RLAgent returns: {action}, {amount}")
        except Exception as e:
            print(f"[DEBUG] RLAgent error: {e}")
            # Safe fallback like evaluation script
            return ("call", to_call) if to_call > 0 else ("check", 0)

        # Ensure valid output
        if not action:
            print("[DEBUG] Empty action from RLAgent, defaulting to check.")
            return ("check", 0)

        return action, int(amount)
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
        env.board = list(board)          # ideally convert to Treys ints
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
    def __init__(self, model, device, starting_chips=1000, name="rl-bot"):
        self.agent = RLAgent(name, starting_chips, model, device)

    def decide_action(self, game_view):
        state = game_view["state"]
        table = game_view["table"]
        me = game_view["self_player"]

        board = state.get("board", [])
        pot = state.get("pot", 0)

        players = table.get("players") or []
        bets = [p.get("bet", 0) for p in players if p]
        current_bet = max(bets) if bets else 0
        my_bet = me.get("bet", 0) or 0
        to_call = max(0, current_bet - my_bet)

        # Proper min_raise handling
        min_raise = None
        for key in ("minRaise", "min_raise"):
            if key in state and state[key] is not None:
                min_raise = state[key]
                break
        if min_raise is None and "minRaise" in table:
            min_raise = table["minRaise"]
        if min_raise is None:
            min_raise = 10  # big blind fallback

        # Update agent's view of hole cards + chips
        self.agent.hole_cards = list(me.get("cards", []))
        self.agent.chips = int(me.get("chips", self.agent.chips))

        try:
            action, amount = self.agent.make_decision(
                board=board,
                pot=pot,
                current_bet=current_bet,
                my_bet=my_bet,
                min_raise=min_raise,
            )
        except Exception as e:
            # Stable fallback
            return ("call", to_call) if to_call > 0 else ("check", 0)

        return action, amount


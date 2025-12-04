"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerAgent to work with TexasHoldemSimulation.
"""

import torch
import numpy as np
from agents.agent import PokerAgent
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import RandomAgent
# Defer importing the model to avoid circular imports (training <-> agents)


class RLAgent(PokerAgent):
    """
    RL agent that uses a trained PPO model to make poker decisions.
    """
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, player_id: str, starting_money: int, device=None):
        """Load RLAgent from a trained checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
            player_id: Name/ID for this agent
            starting_money: Starting chip stack
            device: PyTorch device (defaults to DEVICE from utils.device)
            
        Returns:
            RLAgent instance with loaded model
        """
        import torch
        from utils.device import DEVICE as DEFAULT_DEVICE
        # Import model here to avoid circular imports at module import time
        from training.ppo_model import PokerPPOModel
        
        device = device or DEFAULT_DEVICE
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model with same architecture
        model = PokerPPOModel(
            card_embed_dim=64,
            hidden_dim=256,
            num_shared_layers=2,
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Create and return agent
        return cls(name=player_id, starting_chips=starting_money, model=model, device=device)
    
    def __init__(self, name, starting_chips, model, device):
        """
        Initialize RL agent.
        
        Args:
            name: Agent name
            starting_chips: Starting chip stack
            model: Trained PokerPPOModel
            device: PyTorch device
        """
        super().__init__(name, starting_chips)
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create a temporary environment for generating observations
        # This is just used to format inputs for the model
        config = PokerEnvConfig(
            big_blind=10,
            small_blind=5,
            starting_stack=starting_chips,
            max_players=2,
        )
        placeholder = RandomAgent("Placeholder", starting_chips)
        self.temp_env = PokerEnv(
            players=[placeholder, placeholder],
            config=config,
            hero_idx=0,
        )
    
    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Make a betting decision using the trained RL model.
        
        Args:
            board: Current community cards
            pot_size: Current pot size
            current_bet_to_call: Amount needed to call
            min_raise: Minimum raise amount
            
        Returns:
            Tuple of (action, amount)
        """
        if self.is_folded or self.is_all_in:
            return ('check', 0)
        
        # Sync temporary environment state with current game state
        self._sync_temp_env(board, pot_size, current_bet_to_call, min_raise)
        
        # Get observation from environment
        obs = self.temp_env._get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            fold_logit, bet_alpha, bet_beta, value = self.model(obs_tensor)
            
            # Sample action
            p_fold = torch.sigmoid(fold_logit).item()
            bet_dist = torch.distributions.Beta(bet_alpha, bet_beta)
            bet_scalar = bet_dist.sample().item()
        
        # Interpret action using environment's interpreter
        from agents.poker_player import PokerAction, ActionType
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=self.current_bet + current_bet_to_call,  # Total bet needed
            my_bet=self.current_bet,
            min_raise=min_raise,
            my_money=self.chips,
        )
        
        # Convert PokerAction to (action_str, amount) format
        if poker_action.action_type == ActionType.FOLD:
            return ('fold', 0)
        elif poker_action.action_type == ActionType.CHECK:
            return ('check', 0)
        elif poker_action.action_type == ActionType.CALL:
            return ('call', current_bet_to_call)
        elif poker_action.action_type == ActionType.RAISE:
            return ('raise', poker_action.amount)
        else:
            # Fallback to check
            return ('check', 0)
    
    def _sync_temp_env(self, board, pot_size, current_bet_to_call, min_raise):
        """Synchronize temporary environment with current game state."""
        # Update temp environment's state to match current game
        self.temp_env.board = board[:]
        self.temp_env.pot.money = pot_size
        self.temp_env.current_bet = self.current_bet + current_bet_to_call
        self.temp_env.min_raise = min_raise
        
        # Update hero's state (player 0)
        hero = self.temp_env.players[0]
        hero.hole_cards = self.hole_cards[:]
        hero.money = self.chips
        hero.bet = self.current_bet
        hero.folded = self.is_folded
        hero.all_in = self.is_all_in
        
        # Set round stage based on board size
        if len(board) == 0:
            self.temp_env.round_stage = "pre-flop"
        elif len(board) == 3:
            self.temp_env.round_stage = "flop"
        elif len(board) == 4:
            self.temp_env.round_stage = "turn"
        elif len(board) == 5:
            self.temp_env.round_stage = "river"
    
    def __str__(self):
        """String representation."""
        return f"{self.name} [RL]"


# Backwards-compatible helpers expected by older wrappers
def load_model(checkpoint_path: str):
    """Load a trained model and return (model, device).

    This mirrors the API some older code (e.g. `connect/player/player1.py`) expects.
    """
    import torch
    from utils.device import DEVICE as DEFAULT_DEVICE
    from training.ppo_model import PokerPPOModel

    device = DEFAULT_DEVICE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PokerPPOModel(
        card_embed_dim=64,
        hidden_dim=256,
        num_shared_layers=2,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


class RLBot:
    """Thin compatibility wrapper exposing `decide_action(game_view)`.

    The project previously exported `RLBot` and `load_model`. Newer code uses
    `RLAgent` and `from_checkpoint`. This wrapper lets older callers keep working.
    """

    def __init__(self, model, device, starting_chips: int = 1000, name: str = "rl-bot"):
        # Create an RLAgent instance that will perform the actual decision logic
        self._agent = RLAgent(name=name, starting_chips=starting_chips, model=model, device=device)

    def decide_action(self, game_view: dict):
        """Adapt `game_view` (from connect client) to RLAgent.make_decision inputs.

        game_view expected shape (from `connect/player/player1.py`):
            {
                "state": {"board": [...], "pot": <num>, ...},
                "table": {"players": [...]},
                "self_player": {...},
                "my_seat": int,
            }
        """
        state = game_view.get("state", {})
        table = game_view.get("table", {})
        self_player = game_view.get("self_player") or {}

        board = state.get("board", [])
        pot_size = state.get("pot", 0)

        # Compute current bet to call from player bets if explicit field not present
        players = table.get("players") or []
        bets = [p.get("bet", 0) for p in players if p]
        current_bet = max(bets) if bets else 0
        my_bet = int(self_player.get("bet", 0)) if self_player else 0
        current_bet_to_call = max(0, current_bet - my_bet)

        # Try a few common names for min-raise
        min_raise = state.get("minRaise") or state.get("min_raise") or table.get("minRaise") or 0

        # Populate the internal RLAgent state from the incoming game_view so
        # the model receives the same inputs it expects (hole cards, chips,
        # current bet, folded / all-in flags).
        try:
            agent = self._agent
            # Hole cards: try common keys, leave as-is if absent
            agent.hole_cards = list(self_player.get("cards") or self_player.get("hole_cards") or agent.hole_cards)
            # Chips / money
            try:
                agent.chips = int(self_player.get("chips", agent.chips))
            except Exception:
                agent.chips = agent.chips
            # Current bet (amount already put in this round)
            try:
                agent.current_bet = int(self_player.get("bet", self_player.get("current_bet", agent.current_bet)))
            except Exception:
                agent.current_bet = agent.current_bet
            # Fold / all-in flags (try several possible key names)
            agent.is_folded = bool(self_player.get("folded") or self_player.get("isFolded") or self_player.get("fold", False))
            agent.is_all_in = bool(self_player.get("allIn") or self_player.get("isAllIn") or self_player.get("all_in", False))

            action, amount = agent.make_decision(board, pot_size, current_bet_to_call, min_raise)
        except Exception:
            # On error, fall back to a safe call/check
            return "CALL", 0

        # Return in the shape older callers expect: (action_str, amount)
        return action, amount

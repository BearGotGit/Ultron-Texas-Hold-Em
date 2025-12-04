"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerPlayer and works with both PokerEnv and TexasHoldemSimulation.
"""

import torch
from typing import List, Optional
from pathlib import Path

from agents.poker_player import PokerPlayer, PokerAction, ActionType, PokerPlayerPublic
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import RandomAgent
from training.ppo_model import PokerPPOModel
from utils.device import DEVICE


class RLAgent(PokerPlayer):
    """
    RL agent that uses a trained PPO model to make poker decisions.
    
    This is the canonical RL agent class that works with both:
    - PokerEnv (RL training environment using get_action)
    - TexasHoldemSimulation (interactive game using make_decision)
    """
    
    def __init__(
        self,
        player_id: str,
        starting_money: int,
        model: PokerPPOModel,
        device: torch.device,
        env: Optional[PokerEnv] = None,
    ):
        """
        Initialize RL agent.
        
        Args:
            player_id: Agent name/identifier
            starting_money: Starting chip stack
            model: Trained PokerPPOModel
            device: PyTorch device
            env: Optional PokerEnv for direct observation access
        """
        super().__init__(player_id, starting_money)
        self.model = model
        self.device = device
        self.model.eval()
        self.env = env  # Optional external environment reference
        
        # Create a temporary environment for generating observations
        # when used with TexasHoldemSimulation (which doesn't provide env)
        config = PokerEnvConfig(
            big_blind=10,
            small_blind=5,
            starting_stack=starting_money,
            max_players=2,
        )
        placeholder = RandomAgent("Placeholder", starting_money)
        self._temp_env = PokerEnv(
            players=[placeholder, placeholder],
            config=config,
            hero_idx=0,
        )
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        player_id: str = "RLAgent",
        starting_money: int = 1000,
        device: Optional[torch.device] = None,
    ) -> "RLAgent":
        """
        Load an RLAgent from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            player_id: Agent name/identifier
            starting_money: Starting chip stack
            device: PyTorch device (defaults to utils.device.DEVICE)
            
        Returns:
            RLAgent instance with loaded model
        """
        device = device or DEVICE
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model = PokerPPOModel(
            card_embed_dim=64,
            hidden_dim=256,
            num_shared_layers=2,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        return cls(
            player_id=player_id,
            starting_money=starting_money,
            model=model,
            device=device,
        )
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        """
        Get action from trained RL model (PokerPlayer interface).
        
        Used by PokerEnv for RL training.
        
        Args:
            hole_cards: This player's hole cards (Treys integers)
            board: Community cards (Treys integers)
            pot: Current pot size
            current_bet: Current bet to match
            min_raise: Minimum raise amount
            players: Public info for all players
            my_idx: This player's index in players list
            
        Returns:
            PokerAction describing the chosen action
        """
        # If we have an external environment, use its observation directly
        if self.env is not None:
            obs = self.env._get_observation()
        else:
            # Sync our temp environment and get observation from it
            self._sync_temp_env(hole_cards, board, pot, current_bet, min_raise)
            obs = self._temp_env._get_observation()
        
        # Convert to tensor
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get deterministic action from model
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
        
        action_np = action.squeeze(0).cpu().numpy()
        
        # Interpret action using environment's interpreter
        poker_action = interpret_action(
            p_fold=float(action_np[0]),
            bet_scalar=float(action_np[1]),
            current_bet=current_bet,
            my_bet=self.bet,
            min_raise=min_raise,
            my_money=self.money,
        )
        
        return poker_action
    
    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Make a betting decision (PokerAgent/TexasHoldemSimulation interface).
        
        This method provides compatibility with TexasHoldemSimulation which
        expects the (action_str, amount) tuple format.
        
        Args:
            board: Current community cards
            pot_size: Current pot size
            current_bet_to_call: Amount needed to call
            min_raise: Minimum raise amount
            
        Returns:
            Tuple of (action, amount)
        """
        if self.folded or self.all_in:
            return ('check', 0)
        
        # Sync temporary environment state with current game state
        # Use get_hole_cards() for consistency with the PokerAgent interface
        self._sync_temp_env(
            self.get_hole_cards(),
            board,
            pot_size,
            self.bet + current_bet_to_call,  # Total bet needed
            min_raise
        )
        
        # Get observation from temp environment
        obs = self._temp_env._get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            fold_logit, bet_alpha, bet_beta, value = self.model(obs_tensor)
            
            # Sample action
            p_fold = torch.sigmoid(fold_logit).item()
            bet_dist = torch.distributions.Beta(bet_alpha, bet_beta)
            bet_scalar = bet_dist.sample().item()
        
        # Interpret action using environment's interpreter
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=self.bet + current_bet_to_call,  # Total bet needed
            my_bet=self.bet,
            min_raise=min_raise,
            my_money=self.money,
        )
        
        # Convert PokerAction to (action_str, amount) format for TexasHoldemSimulation
        if poker_action.action_type == ActionType.FOLD:
            return ('fold', 0)
        elif poker_action.action_type == ActionType.CHECK:
            return ('check', 0)
        elif poker_action.action_type == ActionType.CALL:
            return ('call', current_bet_to_call)
        elif poker_action.action_type == ActionType.RAISE:
            raise_amount = poker_action.amount - current_bet_to_call
            return ('raise', raise_amount)
        else:
            # Fallback to check
            return ('check', 0)
    
    def _sync_temp_env(self, hole_cards, board, pot_size, current_bet, min_raise):
        """Synchronize temporary environment with current game state."""
        # Update temp environment's state to match current game
        self._temp_env.board = board[:] if board else []
        self._temp_env.pot.money = pot_size
        self._temp_env.current_bet = current_bet
        self._temp_env.min_raise = min_raise
        
        # Update hero's state (player 0)
        hero = self._temp_env.players[0]
        hero._private_cards = hole_cards[:] if hole_cards else []
        hero.money = self.money
        hero.bet = self.bet
        hero.folded = self.folded
        hero.all_in = self.all_in
        
        # Set round stage based on board size
        board_len = len(board) if board else 0
        if board_len == 0:
            self._temp_env.round_stage = "pre-flop"
        elif board_len == 3:
            self._temp_env.round_stage = "flop"
        elif board_len == 4:
            self._temp_env.round_stage = "turn"
        elif board_len == 5:
            self._temp_env.round_stage = "river"
    
    # Compatibility methods for TexasHoldemSimulation (PokerAgent interface)
    
    def receive_cards(self, cards):
        """Receive hole cards (PokerAgent compatibility)."""
        # Handle both list and tuple input
        if isinstance(cards, (list, tuple)):
            self.deal_hand(tuple(cards))
        else:
            self.deal_hand(cards)
    
    def get_hole_cards(self):
        """Get hole cards (works with both interfaces)."""
        return self._private_cards
    
    def get_chips(self):
        """Get current chip stack (PokerAgent compatibility)."""
        return self.money
    
    def reset_for_new_hand(self):
        """Reset for new hand (PokerAgent compatibility)."""
        self.reset()
    
    def reset_current_bet(self):
        """Reset current bet (PokerAgent compatibility)."""
        self.reset_bet()
    
    @property
    def chips(self):
        """Alias for money (PokerAgent compatibility)."""
        return self.money
    
    @chips.setter
    def chips(self, value):
        """Alias for money setter (PokerAgent compatibility)."""
        self.money = value
    
    @property
    def current_bet(self):
        """Alias for bet (PokerAgent compatibility)."""
        return self.bet
    
    @current_bet.setter
    def current_bet(self, value):
        """Alias for bet setter (PokerAgent compatibility)."""
        self.bet = value
    
    @property
    def is_folded(self):
        """Alias for folded (PokerAgent compatibility)."""
        return self.folded
    
    @is_folded.setter
    def is_folded(self, value):
        """Alias for folded setter (PokerAgent compatibility)."""
        self.folded = value
    
    @property
    def is_all_in(self):
        """Alias for all_in (PokerAgent compatibility)."""
        return self.all_in
    
    @is_all_in.setter
    def is_all_in(self, value):
        """Alias for all_in setter (PokerAgent compatibility)."""
        self.all_in = value
    
    @property
    def hole_cards(self):
        """Alias for _private_cards (PokerAgent compatibility)."""
        return self._private_cards
    
    @hole_cards.setter
    def hole_cards(self, value):
        """Alias for _private_cards setter (PokerAgent compatibility)."""
        self._private_cards = value
    
    @property
    def name(self):
        """Alias for id (PokerAgent compatibility)."""
        return self.id
    
    @name.setter
    def name(self, value):
        """Alias for id setter (PokerAgent compatibility)."""
        self.id = value
    
    def add_chips(self, amount):
        """Add chips to stack (PokerAgent compatibility)."""
        self.add_winnings(amount)
    
    def place_bet(self, amount):
        """
        Place a bet (PokerAgent compatibility).
        
        Matches PokerAgent interface by directly modifying internal state.
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        # Prevent negative bets
        if amount < 0:
            return 0
        
        actual_bet = min(amount, self.money)
        self.money -= actual_bet
        self.bet += actual_bet
        self.total_invested += actual_bet
        
        if self.money == 0:
            self.all_in = True
        
        return actual_bet
    
    def __str__(self):
        """String representation."""
        return f"{self.id} [RL]"

"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerPlayer and works with PokerEnv.
"""

import torch
from typing import List, Optional

from agents.poker_player import PokerPlayer, PokerAction, PokerPlayerPublic
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import RandomAgent
from training.ppo_model import PokerPPOModel
from utils.device import DEVICE


class RLAgent(PokerPlayer):
    """
    RL agent that uses a trained PPO model to make poker decisions.
    
    This is the canonical RL agent class that works with PokerEnv
    for RL training using the get_action interface.
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
    
    def __str__(self):
        """String representation."""
        return f"{self.id} [RL]"

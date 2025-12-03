"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerAgent to work with TexasHoldemSimulation.
"""

import torch
import numpy as np
from agents.agent import PokerAgent
from simulation.poker_env import PokerEnv, PokerEnvConfig, interpret_action
from agents.monte_carlo_agent import RandomAgent
from training.ppo_model import PokerPPOModel


class RLAgent(PokerAgent):
    """
    RL agent that uses a trained PPO model to make poker decisions.
    """
    
    def __init__(self, name, starting_chips, model: PokerPPOModel, device):
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
            raise_amount = poker_action.amount - current_bet_to_call
            return ('raise', raise_amount)
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

"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerAgent to work with TexasHoldemSimulation.
"""

from typing import List
import torch
import numpy as np
from agents.agent import PokerAgent
from agents.poker_player import PokerPlayerPublic, PokerAction, ActionType
from simulation.poker_env import (
    PokerEnv,
    PokerEnvConfig,
    interpret_action,
    encode_card_one_hot,
    encode_hand_features,
    encode_round_stage,
    MAX_PLAYERS,
    FEATURES_PER_PLAYER,
    GLOBAL_NUMERIC_FEATURES,
)
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
        Decide on an action given the game state.
        
        Implements the PokerPlayer.get_action() interface for compatibility
        with PokerEnv and other systems that use the PokerPlayer interface.
        
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
        my_info = players[my_idx]
        
        # Can't act if folded or all-in
        if my_info.folded or my_info.all_in:
            return PokerAction.check()
        
        # Build observation from game state
        obs = self._build_observation(
            hole_cards=hole_cards,
            board=board,
            pot=pot,
            current_bet=current_bet,
            min_raise=min_raise,
            players=players,
            my_idx=my_idx,
        )
        
        # Convert to tensor and move to device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            fold_logit, bet_alpha, bet_beta, value = self.model(obs_tensor)
            
            # Sample action
            p_fold = torch.sigmoid(fold_logit).item()
            bet_dist = torch.distributions.Beta(bet_alpha, bet_beta)
            bet_scalar = bet_dist.sample().item()
        
        # Interpret model output into PokerAction
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=current_bet,
            my_bet=my_info.bet,
            min_raise=min_raise,
            my_money=my_info.money,
        )
        
        return poker_action
    
    def _build_observation(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> np.ndarray:
        """
        Build observation tensor from game state.
        
        Reuses the observation encoding logic from PokerEnv to ensure
        consistency between training and inference.
        
        Args:
            hole_cards: This player's hole cards (Treys integers)
            board: Community cards (Treys integers)
            pot: Current pot size
            current_bet: Current bet to match
            min_raise: Minimum raise amount
            players: Public info for all players
            my_idx: This player's index in players list
            
        Returns:
            Observation tensor as numpy array
        """
        obs_parts = []
        
        # 1. Card encodings (7 x 53)
        # Hole cards (2 slots)
        for i in range(2):
            card = hole_cards[i] if i < len(hole_cards) else None
            obs_parts.append(encode_card_one_hot(card))
        
        # Board cards (5 slots)
        for i in range(5):
            card = board[i] if i < len(board) else None
            obs_parts.append(encode_card_one_hot(card))
        
        # 2. Hand features (10 binary flags)
        hand_features = encode_hand_features(hole_cards, board)
        obs_parts.append(hand_features)
        
        # 3. Player features (MAX_PLAYERS x 4)
        # Use config values for normalization
        starting_stack = self.temp_env.config.starting_stack
        big_blind = self.temp_env.config.big_blind
        
        player_features_dim = MAX_PLAYERS * FEATURES_PER_PLAYER
        player_features = np.zeros(player_features_dim, dtype=np.float32)
        
        for i, player in enumerate(players):
            if i >= MAX_PLAYERS:
                break
            base = i * FEATURES_PER_PLAYER
            # Normalize money: log(money+1) / log(starting_stack+1)
            stack_normalizer = np.log1p(starting_stack)
            player_features[base] = np.log1p(player.money) / stack_normalizer
            # Normalize bet: log(bet+1) / log(big_blind+1)
            bb_normalizer = np.log1p(big_blind) if big_blind > 0 else 1.0
            player_features[base + 1] = np.log1p(player.bet) / bb_normalizer
            # Binary flags
            player_features[base + 2] = float(player.folded)
            player_features[base + 3] = float(player.all_in)
        obs_parts.append(player_features)
        
        # 4. Global features
        # Determine round stage from board size
        if len(board) == 0:
            round_stage = "pre-flop"
        elif len(board) == 3:
            round_stage = "flop"
        elif len(board) == 4:
            round_stage = "turn"
        else:
            round_stage = "river"
        
        # Normalizers
        num_players = len(players)
        total_starting_money = starting_stack * num_players
        stack_normalizer = np.log1p(total_starting_money)
        bb_normalizer = np.log1p(big_blind) if big_blind > 0 else 1.0
        
        # Dealer position - default to 0 since we don't have this info
        # in the PokerPlayer.get_action() interface
        dealer_position = 0
        
        global_features = np.array([
            # Pot: log(pot+1) / log(total_starting_money+1)
            np.log1p(pot) / stack_normalizer,
            # Current bet: log(bet+1) / log(big_blind+1)
            np.log1p(current_bet) / bb_normalizer,
            # Min raise: log(raise+1) / log(big_blind+1)
            np.log1p(min_raise) / bb_normalizer,
            # Round stage: normalize to [0, 1]
            encode_round_stage(round_stage) / 4.0,
            # Hero position: normalize to [0, 1]
            my_idx / max(num_players - 1, 1),
            # Dealer position: normalize to [0, 1]
            dealer_position / max(num_players - 1, 1),
        ], dtype=np.float32)
        obs_parts.append(global_features)
        
        return np.concatenate(obs_parts)
    
    def __str__(self):
        """String representation."""
        return f"{self.name} [RL]"

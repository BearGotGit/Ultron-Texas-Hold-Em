"""
RL Agent wrapper for playing poker using trained PPO model.
Inherits from PokerPlayer and works with PokerEnv.
"""

from typing import List
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
        # Use config values for normalization (calculated once before loop)
        starting_stack = self.temp_env.config.starting_stack
        big_blind = self.temp_env.config.big_blind
        stack_normalizer = np.log1p(starting_stack)
        bb_normalizer = np.log1p(big_blind) if big_blind > 0 else 1.0
        
        player_features = np.zeros(MAX_PLAYERS * FEATURES_PER_PLAYER, dtype=np.float32)
        
        for i, player in enumerate(players):
            if i >= MAX_PLAYERS:
                break
            base = i * FEATURES_PER_PLAYER
            # Normalize money: log(money+1) / log(starting_stack+1)
            player_features[base] = np.log1p(player.money) / stack_normalizer
            # Normalize bet: log(bet+1) / log(big_blind+1)
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
        # Use config values for normalization (calculated once before loop)
        starting_stack = self.temp_env.config.starting_stack
        big_blind = self.temp_env.config.big_blind
        stack_normalizer = np.log1p(starting_stack)
        bb_normalizer = np.log1p(big_blind) if big_blind > 0 else 1.0
        
        player_features = np.zeros(MAX_PLAYERS * FEATURES_PER_PLAYER, dtype=np.float32)
        
        for i, player in enumerate(players):
            if i >= MAX_PLAYERS:
                break
            base = i * FEATURES_PER_PLAYER
            # Normalize money: log(money+1) / log(starting_stack+1)
            player_features[base] = np.log1p(player.money) / stack_normalizer
            # Normalize bet: log(bet+1) / log(big_blind+1)
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
        return f"{self.id} [RL]"

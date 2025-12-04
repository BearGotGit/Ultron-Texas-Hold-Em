"""
PPO Model for Texas Hold'em.

Architecture:
    - Card embedding block: 53-dim one-hot → 64 → 64 (shared across 7 cards)
    - Hand embedding block: 10 binary flags → 32 → 32
    - Numeric embedding block: 42 features → 64 → 64
    - Combined features: 448 (cards) + 32 (hands) + 64 (numeric) = 544
    - Shared dense layers: 544 → 256 → 256
    - Three heads:
        1. Fold head: outputs logit for p_fold (Bernoulli)
        2. Bet head: outputs alpha and beta for bet_scalar (Beta distribution)
        3. Value head: outputs V(s) estimate for critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta
from typing import Tuple, Optional

# Import observation constants
from simulation.poker_env import (
    CARD_ENCODING_DIM,
    NUM_CARD_SLOTS,
    NUM_HAND_FEATURES,
    MAX_PLAYERS,
    FEATURES_PER_PLAYER,
    GLOBAL_NUMERIC_FEATURES,
)


class CardEmbedding(nn.Module):
    """
    Embedding block for individual cards.
    Takes a 53-dim one-hot vector and produces a H-dim embedding.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CARD_ENCODING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 53) one-hot card encoding
            
        Returns:
            (batch, hidden_dim) embedding
        """
        return self.net(x)


class PokerPPOModel(nn.Module):
    """
    PPO Actor-Critic model for Texas Hold'em.
    
    Observation space (flat vector):
        - 7 x 53 card one-hot encodings (hole cards + board)
        - 10 binary hand features
        - MAX_PLAYERS x 4 player state features
        - 6 global numeric features
        
    Action space:
        - p_fold: Bernoulli probability of folding
        - bet_scalar: Beta distribution parameter for bet sizing
    """
    
    def __init__(
        self,
        card_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_shared_layers: int = 2,
    ):
        super().__init__()
        
        self.card_embed_dim = card_embed_dim
        self.hidden_dim = hidden_dim
        
        # Card embedding (shared across all 7 card slots)
        self.card_embedding = CardEmbedding(card_embed_dim)
        
        # Hand feature embedding (10 binary flags -> 32-dim embedding)
        self.hand_embedding = nn.Sequential(
            nn.Linear(NUM_HAND_FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        # Numeric feature embedding (player + global features -> 64-dim embedding)
        numeric_input_dim = MAX_PLAYERS * FEATURES_PER_PLAYER + GLOBAL_NUMERIC_FEATURES  # 36 + 6 = 42
        self.numeric_embedding = nn.Sequential(
            nn.Linear(numeric_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Calculate input dimensions
        self.card_features = NUM_CARD_SLOTS * card_embed_dim  # 7 * 64 = 448
        self.hand_embed_features = 32  # embedded hand features
        self.numeric_embed_features = 64  # embedded numeric features
        
        self.combined_features = (
            self.card_features +
            self.hand_embed_features +
            self.numeric_embed_features
        )  # 448 + 32 + 64 = 544
        
        # Shared trunk layers
        shared_layers = []
        input_dim = self.combined_features
        for _ in range(num_shared_layers):
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        self.shared_trunk = nn.Sequential(*shared_layers)
        
        # Fold head (Bernoulli logit)
        self.fold_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # logit for p_fold
        )
        
        # Bet head (Beta distribution parameters)
        # Beta(alpha, beta) where alpha, beta > 0
        self.bet_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # alpha, beta for Beta distribution
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization.
        
        Uses smaller gains for output heads to prevent saturation of
        fold probabilities at 0 or 1 during early training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Use smaller initialization for the fold head final layer
        # to prevent saturation (extreme logits → p_fold ≈ 0 or 1)
        # A gain of 0.01 keeps initial logits small (~0) → p_fold ≈ 0.5
        # Note: fold_head and bet_head are nn.Sequential, so [-1] gets the last Linear layer
        fold_final_layer = self.fold_head[-1]
        nn.init.orthogonal_(fold_final_layer.weight, gain=0.01)
        nn.init.zeros_(fold_final_layer.bias)
        
        # Similarly, use smaller initialization for bet head final layer
        # to produce more uniform Beta distribution parameters initially
        bet_final_layer = self.bet_head[-1]
        nn.init.orthogonal_(bet_final_layer.weight, gain=0.01)
        nn.init.zeros_(bet_final_layer.bias)
    
    def _embed_cards(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract and embed card observations.
        
        Args:
            obs: (batch, obs_dim) full observation tensor
            
        Returns:
            (batch, card_features) embedded card representations
        """
        batch_size = obs.shape[0]
        card_obs_dim = NUM_CARD_SLOTS * CARD_ENCODING_DIM  # 7 * 53 = 371
        
        # Extract card one-hots
        card_obs = obs[:, :card_obs_dim]  # (batch, 371)
        
        # Reshape to (batch, 7, 53)
        card_obs = card_obs.view(batch_size, NUM_CARD_SLOTS, CARD_ENCODING_DIM)
        
        # Embed each card slot
        card_embeddings = []
        for i in range(NUM_CARD_SLOTS):
            card_emb = self.card_embedding(card_obs[:, i, :])  # (batch, card_embed_dim)
            card_embeddings.append(card_emb)
        
        # Concatenate: (batch, 7 * card_embed_dim)
        return torch.cat(card_embeddings, dim=1)
    
    def _extract_hand_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract hand features from observation.
        
        Args:
            obs: (batch, obs_dim) full observation tensor
            
        Returns:
            (batch, NUM_HAND_FEATURES) hand features tensor
        """
        card_obs_dim = NUM_CARD_SLOTS * CARD_ENCODING_DIM  # 371
        hand_start = card_obs_dim
        hand_end = hand_start + NUM_HAND_FEATURES  # 371 + 10 = 381
        
        return obs[:, hand_start:hand_end]  # (batch, 10)
    
    def _extract_numeric_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract numeric features (player + global) from observation.
        
        Args:
            obs: (batch, obs_dim) full observation tensor
            
        Returns:
            (batch, numeric_features) numeric features tensor
        """
        card_obs_dim = NUM_CARD_SLOTS * CARD_ENCODING_DIM  # 371
        numeric_start = card_obs_dim + NUM_HAND_FEATURES  # 371 + 10 = 381
        
        # Player features + global features
        return obs[:, numeric_start:]  # (batch, 36 + 6 = 42)
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: (batch, obs_dim) observation tensor
            
        Returns:
            fold_logit: (batch, 1) logit for fold probability
            bet_alpha: (batch, 1) alpha parameter for Beta distribution
            bet_beta: (batch, 1) beta parameter for Beta distribution
            value: (batch, 1) state value estimate
        """
        # Embed cards
        card_features = self._embed_cards(obs)  # (batch, 448)
        
        # Extract and embed hand features
        hand_raw = self._extract_hand_features(obs)  # (batch, 10)
        hand_features = self.hand_embedding(hand_raw)  # (batch, 32)
        
        # Extract and embed numeric features
        numeric_raw = self._extract_numeric_features(obs)  # (batch, 42)
        numeric_features = self.numeric_embedding(numeric_raw)  # (batch, 64)
        
        # Concatenate all embedded features
        combined = torch.cat([card_features, hand_features, numeric_features], dim=1)  # (batch, 544)
        
        # Shared trunk
        hidden = self.shared_trunk(combined)  # (batch, hidden_dim)
        
        # Fold head
        fold_logit = self.fold_head(hidden)  # (batch, 1)
        
        # Bet head (ensure positive alpha, beta via softplus)
        bet_params = self.bet_head(hidden)  # (batch, 2)
        bet_alpha = F.softplus(bet_params[:, 0:1]) + 1.0  # (batch, 1), min 1.0
        bet_beta = F.softplus(bet_params[:, 1:2]) + 1.0  # (batch, 1), min 1.0
        
        # Value head
        value = self.value_head(hidden)  # (batch, 1)
        
        return fold_logit, bet_alpha, bet_beta, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value/log_probs for PPO.
        
        Args:
            obs: (batch, obs_dim) observation tensor
            deterministic: If True, return mean action instead of sampling
            
        Returns:
            action: (batch, 2) [p_fold, bet_scalar]
            log_prob: (batch,) log probability of the action
            entropy: (batch,) action entropy
            value: (batch, 1) state value
        """
        fold_logit, bet_alpha, bet_beta, value = self.forward(obs)
        
        # Fold distribution (Bernoulli)
        fold_prob = torch.sigmoid(fold_logit)
        fold_dist = Bernoulli(probs=fold_prob)
        
        # Bet distribution (Beta)
        bet_dist = Beta(bet_alpha, bet_beta)
        
        if deterministic:
            # Use mode/mean for deterministic action
            fold_action = (fold_prob > 0.5).float()
            bet_action = bet_dist.mean
        else:
            # Sample
            fold_action = fold_dist.sample()
            bet_action = bet_dist.sample()
        
        # Combine into action tensor
        action = torch.cat([fold_action, bet_action], dim=1)  # (batch, 2)
        
        # Log probabilities
        fold_log_prob = fold_dist.log_prob(fold_action)  # (batch, 1)
        bet_log_prob = bet_dist.log_prob(bet_action)  # (batch, 1)
        log_prob = fold_log_prob.squeeze(-1) + bet_log_prob.squeeze(-1)  # (batch,)
        
        # Entropy
        fold_entropy = fold_dist.entropy()  # (batch, 1)
        bet_entropy = bet_dist.entropy()  # (batch, 1)
        entropy = fold_entropy.squeeze(-1) + bet_entropy.squeeze(-1)  # (batch,)
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given actions.
        Used during PPO update.
        
        Args:
            obs: (batch, obs_dim) observations
            actions: (batch, 2) [p_fold, bet_scalar] actions
            
        Returns:
            log_prob: (batch,) log probability of actions
            entropy: (batch,) action entropy
            value: (batch, 1) state values
        """
        fold_logit, bet_alpha, bet_beta, value = self.forward(obs)
        
        # Extract action components
        fold_action = actions[:, 0:1]  # (batch, 1)
        bet_action = actions[:, 1:2]  # (batch, 1)
        
        # Clamp bet_action to valid range for Beta distribution
        bet_action = torch.clamp(bet_action, 1e-6, 1.0 - 1e-6)
        
        # Fold distribution
        fold_prob = torch.sigmoid(fold_logit)
        fold_dist = Bernoulli(probs=fold_prob)
        
        # Bet distribution
        bet_dist = Beta(bet_alpha, bet_beta)
        
        # Log probabilities
        fold_log_prob = fold_dist.log_prob(fold_action)  # (batch, 1)
        bet_log_prob = bet_dist.log_prob(bet_action)  # (batch, 1)
        log_prob = fold_log_prob.squeeze(-1) + bet_log_prob.squeeze(-1)  # (batch,)
        
        # Entropy
        fold_entropy = fold_dist.entropy()  # (batch, 1)
        bet_entropy = bet_dist.entropy()  # (batch, 1)
        entropy = fold_entropy.squeeze(-1) + bet_entropy.squeeze(-1)  # (batch,)
        
        return log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value only (for bootstrapping).
        
        Args:
            obs: (batch, obs_dim) observation tensor
            
        Returns:
            value: (batch, 1) state value
        """
        _, _, _, value = self.forward(obs)
        return value


class RLPokerPlayer:
    """
    Wrapper to use PokerPPOModel as a PokerPlayer for self-play.
    """
    
    def __init__(
        self,
        player_id: str,
        model: PokerPPOModel,
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ):
        self.id = player_id
        self.model = model
        self.device = device
        self.deterministic = deterministic
    
    def get_action_from_obs(self, obs: torch.Tensor) -> Tuple[float, float]:
        """
        Get action from observation tensor.
        
        Returns:
            (p_fold, bet_scalar) tuple
        """
        self.model.eval()
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(self.device)  # (1, obs_dim)
            action, _, _, _ = self.model.get_action_and_value(
                obs, deterministic=self.deterministic
            )
            action = action.squeeze(0).cpu().numpy()
        return float(action[0]), float(action[1])


if __name__ == "__main__":
    # Test the model
    from simulation.poker_env import PokerEnv, PokerEnvConfig
    
    # Create dummy observation
    obs_dim = (
        NUM_CARD_SLOTS * CARD_ENCODING_DIM +
        NUM_HAND_FEATURES +
        MAX_PLAYERS * FEATURES_PER_PLAYER +
        GLOBAL_NUMERIC_FEATURES
    )
    print(f"Observation dimension: {obs_dim}")
    
    # Create model
    model = PokerPPOModel()
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 8
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs)
    print(f"\nTest forward pass:")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Value shape: {value.shape}")
    
    # Test evaluate_actions
    log_prob2, entropy2, value2 = model.evaluate_actions(dummy_obs, action)
    print(f"\nTest evaluate_actions:")
    print(f"  Log prob shape: {log_prob2.shape}")
    print(f"  Entropy shape: {entropy2.shape}")
    print(f"  Value shape: {value2.shape}")
    
    print("\n✓ Model tests passed!")

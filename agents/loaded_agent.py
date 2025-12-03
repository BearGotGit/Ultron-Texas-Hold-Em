# agents/ppo_player.py

from __future__ import annotations

from os import name
from typing import List, Optional
from xml.etree.ElementTree import tostring

import numpy as np
import torch

from agents import agent
from agents.agent import PokerAgent
from agents.poker_player import PokerPlayer, PokerAction
from training.ppo_model import PokerPPOModel
from simulation.poker_env import (
    CARD_ENCODING_DIM,
    NUM_CARD_SLOTS,
    NUM_HAND_FEATURES,
    MAX_PLAYERS,
    FEATURES_PER_PLAYER,
    encode_card_one_hot,
    encode_hand_features,
    encode_round_stage,
    interpret_action,
)
from utils.device import DEVICE
import torch.serialization

import __main__
# Import PPOConfig from the trainer module where it's defined
from training.train_rl_model import PPOConfig as TrainerPPOConfig # adjust path if needed
 
__main__.PPOConfig = TrainerPPOConfig

# Allowlist PPOConfig so safe loader can unpickle it
torch.serialization.add_safe_globals([__main__.PPOConfig])

class LoadAgentPlayer(PokerAgent):
    """
    PokerPlayer that uses a trained PokerPPOModel to choose actions.

    This is designed to plug into the same interface as your existing agents:

        get_action(
            hole_cards,
            board,
            pot,
            current_bet,
            min_raise,
            players,
            my_idx,
        )

    Example usage:

        from agents.ppo_player import PPOPokerPlayer

        ai_player = PPOPokerPlayer(
            name="AI-Hero",
            starting_chips=1000,
            checkpoint_path="checkpoints/final.pt",
        )
    """

    def __init__(
        self,
        name: str,
        starting_chips: int,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        # Optional overrides (will default to values from checkpoint config)
        starting_stack: Optional[int] = None,
        big_blind: Optional[int] = None,
        max_players: int = MAX_PLAYERS,
    ):
        super().__init__(name, starting_chips)

        self.device = device or DEVICE
        self.players = None
        self.AI_index = 0

        # ---- Load checkpoint and config ----
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]  # PPOConfig

        # Use training-time hyperparameters unless explicitly overridden
        self.starting_stack = starting_stack if starting_stack is not None else cfg.starting_stack
        self.big_blind = big_blind if big_blind is not None else cfg.big_blind
        self.max_players = max_players

        # Number of players model expects (from training)
        self.trained_num_players: int = cfg.num_players

        # ---- Recreate model exactly like in training ----
        self.model = PokerPPOModel(
            card_embed_dim=cfg.card_embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_shared_layers=cfg.num_shared_layers,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Useful for normalization; will be updated per hand if needed
        self.current_num_players: int = self.trained_num_players

        # You can optionally update this from your game engine if you track dealer rotation
        self.dealer_position: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_dealer_position(self, dealer_pos: int) -> None:
        """
        Optionally call this from your game engine at the start of each hand
        if you want the observation to match training more closely.
        """
        self.dealer_position = dealer_pos

    @staticmethod
    def _infer_round_stage_from_board(board: List[int]) -> str:
        """
        Infer round stage ("pre-flop", "flop", "turn", "river") from board length.

        This matches how your environment uses round_stage for encoding:
            0: pre-flop
            1: flop
            2: turn
            3: river
            4: showdown  (not needed for action decisions)
        """
        n = len(board)
        if n == 0:
            return "pre-flop"
        elif n == 3:
            return "flop"
        elif n == 4:
            return "turn"
        elif n == 5:
            return "river"
        else:
            # Fallback; shouldn't normally happen when deciding actions
            return "pre-flop"
    
    def _build_observation(
        self,
        hole_cards,
        board,
        pot,
        current_bet,
        min_raise,
        players,
        my_idx: int,
    ) -> torch.Tensor:
        """
        Build an observation vector IDENTICAL to PokerEnv._get_observation().

        Args mirror PokerPlayer.get_action, but we:
            - Treat `players` as a list of objects with attributes:
              money, bet, folded, all_in
            - Use `hole_cards` for the hero (this player).

        Returns:
            obs_t: torch.Tensor of shape [1, obs_dim] on self.device
        """
        # Make sure num_players matches current table
        num_players = len(players)
        self.current_num_players = num_players

        # 1. Card encodings (7 x 53)
        obs_parts = []

        # Hero hole cards (2 slots)
        for i in range(2):
            card = hole_cards[i] if i < len(hole_cards) else None
            obs_parts.append(encode_card_one_hot(card))

        # Board cards (5 slots)
        for i in range(5):
            card = board[i] if i < len(board) else None
            obs_parts.append(encode_card_one_hot(card))

        # 2. Hand features (10 binary flags)
        hand_features = encode_hand_features(list(hole_cards), list(board))
        obs_parts.append(hand_features)

        # 3. Player features (MAX_PLAYERS x 4)
        player_features_dim = self.max_players * FEATURES_PER_PLAYER
        player_features = np.zeros(player_features_dim, dtype=np.float32)

        stack_normalizer = np.log1p(self.starting_stack)
        bb_normalizer = np.log1p(self.big_blind) if self.big_blind > 0 else 1.0

        for i, p in enumerate(players):
            if i >= self.max_players:
                break
            base = i * FEATURES_PER_PLAYER

            # Expect public objects with attributes money, bet, folded, all_in
            money = getattr(p, "money", 0)
            bet = getattr(p, "bet", 0)
            folded = getattr(p, "folded", False)
            all_in = getattr(p, "all_in", False)

            # money: log(money+1) / log(starting_stack+1)
            player_features[base] = np.log1p(money) / stack_normalizer

            # bet: log(bet+1) / log(big_blind+1)
            player_features[base + 1] = np.log1p(bet) / bb_normalizer

            # folded / all_in flags
            player_features[base + 2] = float(bool(folded))
            player_features[base + 3] = float(bool(all_in))

        obs_parts.append(player_features)

        # 4. Global features (6)
        total_starting_chips = self.starting_stack * num_players
        stack_normalizer_total = np.log1p(total_starting_chips)
        bb_normalizer_total = np.log1p(self.big_blind) if self.big_blind > 0 else 1.0

        round_stage = self._infer_round_stage_from_board(board)

        global_features = np.array([
            # Pot: log(pot+1) / log(total_starting_chips+1)
            np.log1p(pot) / stack_normalizer_total,
            # Current bet: log(bet+1) / log(big_blind+1)
            np.log1p(current_bet) / bb_normalizer_total,
            # Min raise: log(raise+1) / log(big_blind+1)
            np.log1p(min_raise) / bb_normalizer_total,
            # Round stage: normalize to [0, 1]
            encode_round_stage(round_stage) / 4.0,
            # Hero position: normalize to [0, 1]
            my_idx / max(num_players - 1, 1),
            # Dealer position: normalize to [0, 1]
            self.dealer_position / max(num_players - 1, 1),
        ], dtype=np.float32)

        obs_parts.append(global_features)

        obs_np = np.concatenate(obs_parts).astype(np.float32)

        # [1, obs_dim] tensor
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_t

    # ------------------------------------------------------------------
    # Main interface: what your engine calls
    # ------------------------------------------------------------------

    def get_action(
        self,
        hole_cards,
        board,
        pot,
        current_bet,
        min_raise,
        players,
        my_idx: int,
    ) -> PokerAction:
        """
        Decide an action for this player, using the trained PPO policy.

        Parameters match the PokerPlayer interface as used in PokerEnv:

            hole_cards: hero's hole cards (Treys int list/tuple)
            board: community cards (Treys int list)
            pot: total pot (int chips)
            current_bet: current bet to match (int chips)
            min_raise: minimum legal raise amount (int chips)
            players: list of PokerPlayerPublic or PokerPlayer-like objects
            my_idx: index of this player in the players list

        Returns:
            PokerAction: one of FOLD/CHECK/CALL/RAISE with an amount.
        """
        # 1. Build observation
        print("Calling Observation")
        obs_t = self._build_observation(
            hole_cards=hole_cards,
            board=board,
            pot=pot,
            current_bet=current_bet,
            min_raise=min_raise,
            players=players,
            my_idx=my_idx,
        )

        # 2. Run model (deterministic for "greedy" play)
        with torch.no_grad():
            action_t, _, _, _ = self.model.get_action_and_value(
                obs_t,
                deterministic=True,
            )

        action_np = action_t.squeeze(0).cpu().numpy()
        p_fold = float(action_np[0])      # 0 or 1 (Bernoulli-sampled in training, but here deterministic)
        bet_scalar = float(action_np[1])  # continuous [0, 1]

        # 3. Use interpret_action to convert into PokerAction
        me_pub = players[my_idx]
        my_bet = getattr(me_pub, "bet", 0)
        my_money = getattr(me_pub, "money", 0)

        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=current_bet,
            my_bet=my_bet,
            min_raise=min_raise,
            my_money=my_money,
        )

        return poker_action

    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        pa : PokerAction= self.get_action( self.hole_cards, board, pot_size, current_bet_to_call, min_raise, self.players, self.AI_index)
        print(f"{self.name} {pa.action_type.value, pa.amount}")
        return (pa.action_type.value,pa.amount)

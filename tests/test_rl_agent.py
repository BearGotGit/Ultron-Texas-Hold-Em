"""
Tests for RLAgent.get_action() interface.

These tests verify that the RLAgent properly implements the PokerPlayer.get_action()
interface, correctly building observations and returning valid PokerActions.
"""

import pytest
import torch
import numpy as np
from treys import Deck, Card

from agents.rl_agent import RLAgent
from agents.poker_player import PokerPlayerPublic, PokerAction, ActionType
from training.ppo_model import PokerPPOModel


@pytest.fixture
def model():
    """Create a fresh model instance for testing."""
    return PokerPPOModel()


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def rl_agent(model, device):
    """Create an RLAgent instance for testing."""
    return RLAgent(
        name="TestRLAgent",
        starting_chips=1000,
        model=model,
        device=device,
    )


@pytest.fixture
def sample_game_state():
    """Create a sample game state for testing."""
    deck = Deck()
    hole_cards = deck.draw(2)
    board = deck.draw(3)  # Flop
    
    # Create player public info
    players = [
        PokerPlayerPublic(
            id="Player0",
            money=950,
            folded=False,
            all_in=False,
            bet=50,
        ),
        PokerPlayerPublic(
            id="Player1",
            money=900,
            folded=False,
            all_in=False,
            bet=100,
        ),
    ]
    
    return {
        "hole_cards": hole_cards,
        "board": board,
        "pot": 150,
        "current_bet": 100,
        "min_raise": 50,
        "players": players,
        "my_idx": 0,
    }


class TestRLAgentGetAction:
    """Tests for RLAgent.get_action() method."""
    
    def test_get_action_returns_poker_action(self, rl_agent, sample_game_state):
        """
        GIVEN an RLAgent and a game state
        WHEN get_action() is called
        THEN it should return a PokerAction instance
        """
        action = rl_agent.get_action(**sample_game_state)
        
        assert isinstance(action, PokerAction)
        assert isinstance(action.action_type, ActionType)
        assert isinstance(action.amount, int)
    
    def test_get_action_returns_valid_action_type(self, rl_agent, sample_game_state):
        """
        GIVEN an RLAgent and a game state
        WHEN get_action() is called multiple times
        THEN it should return valid action types (fold, check, call, raise)
        """
        valid_types = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.RAISE}
        
        # Run multiple times to check different possible outputs
        for _ in range(10):
            action = rl_agent.get_action(**sample_game_state)
            assert action.action_type in valid_types
    
    def test_get_action_handles_preflop(self, rl_agent):
        """
        GIVEN an RLAgent and a pre-flop game state (no board cards)
        WHEN get_action() is called
        THEN it should return a valid PokerAction
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        
        players = [
            PokerPlayerPublic(id="Player0", money=990, folded=False, all_in=False, bet=10),
            PokerPlayerPublic(id="Player1", money=980, folded=False, all_in=False, bet=20),
        ]
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=[],  # Pre-flop
            pot=30,
            current_bet=20,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(action, PokerAction)
    
    def test_get_action_handles_river(self, rl_agent):
        """
        GIVEN an RLAgent and a river game state (5 board cards)
        WHEN get_action() is called
        THEN it should return a valid PokerAction
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(5)  # River
        
        players = [
            PokerPlayerPublic(id="Player0", money=500, folded=False, all_in=False, bet=200),
            PokerPlayerPublic(id="Player1", money=400, folded=False, all_in=False, bet=300),
        ]
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=500,
            current_bet=300,
            min_raise=100,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(action, PokerAction)
    
    def test_get_action_when_folded_returns_check(self, rl_agent):
        """
        GIVEN an RLAgent where the player is folded
        WHEN get_action() is called
        THEN it should return a check action
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(3)
        
        players = [
            PokerPlayerPublic(id="Player0", money=500, folded=True, all_in=False, bet=0),
            PokerPlayerPublic(id="Player1", money=400, folded=False, all_in=False, bet=100),
        ]
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=100,
            current_bet=100,
            min_raise=50,
            players=players,
            my_idx=0,
        )
        
        assert action.action_type == ActionType.CHECK
    
    def test_get_action_when_all_in_returns_check(self, rl_agent):
        """
        GIVEN an RLAgent where the player is all-in
        WHEN get_action() is called
        THEN it should return a check action
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(3)
        
        players = [
            PokerPlayerPublic(id="Player0", money=0, folded=False, all_in=True, bet=1000),
            PokerPlayerPublic(id="Player1", money=500, folded=False, all_in=False, bet=500),
        ]
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=1500,
            current_bet=1000,
            min_raise=100,
            players=players,
            my_idx=0,
        )
        
        assert action.action_type == ActionType.CHECK
    
    def test_get_action_with_multiple_players(self, rl_agent):
        """
        GIVEN an RLAgent and a game with multiple players
        WHEN get_action() is called
        THEN it should return a valid PokerAction
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(4)  # Turn
        
        players = [
            PokerPlayerPublic(id="Player0", money=800, folded=False, all_in=False, bet=100),
            PokerPlayerPublic(id="Player1", money=700, folded=False, all_in=False, bet=100),
            PokerPlayerPublic(id="Player2", money=600, folded=True, all_in=False, bet=50),
            PokerPlayerPublic(id="Player3", money=500, folded=False, all_in=False, bet=150),
        ]
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=400,
            current_bet=150,
            min_raise=50,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(action, PokerAction)


class TestRLAgentBuildObservation:
    """Tests for RLAgent._build_observation() method."""
    
    def test_build_observation_returns_correct_shape(self, rl_agent, sample_game_state):
        """
        GIVEN an RLAgent and a game state
        WHEN _build_observation() is called
        THEN it should return an array with the correct shape
        """
        obs = rl_agent._build_observation(**sample_game_state)
        
        # Expected observation dimension from PokerEnv:
        # Card encodings: 7 * 53 = 371
        # Hand features: 10
        # Player features: 9 * 4 = 36
        # Global features: 6
        # Total: 371 + 10 + 36 + 6 = 423
        expected_dim = 423
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (expected_dim,)
    
    def test_build_observation_returns_float32(self, rl_agent, sample_game_state):
        """
        GIVEN an RLAgent and a game state
        WHEN _build_observation() is called
        THEN it should return a float32 array
        """
        obs = rl_agent._build_observation(**sample_game_state)
        
        assert obs.dtype == np.float32
    
    def test_build_observation_handles_empty_board(self, rl_agent):
        """
        GIVEN an RLAgent and a pre-flop game state
        WHEN _build_observation() is called
        THEN it should handle empty board correctly
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        
        players = [
            PokerPlayerPublic(id="Player0", money=990, folded=False, all_in=False, bet=10),
            PokerPlayerPublic(id="Player1", money=980, folded=False, all_in=False, bet=20),
        ]
        
        obs = rl_agent._build_observation(
            hole_cards=hole_cards,
            board=[],
            pot=30,
            current_bet=20,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(obs, np.ndarray)
        # The observation should be all finite numbers
        assert np.all(np.isfinite(obs))


class TestRLAgentDeviceHandling:
    """Tests for device handling in RLAgent."""
    
    def test_agent_uses_cpu_device(self, model):
        """
        GIVEN a CPU device
        WHEN RLAgent is created
        THEN it should work correctly on CPU
        """
        device = torch.device("cpu")
        agent = RLAgent("CPUAgent", 1000, model, device)
        
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(3)
        
        players = [
            PokerPlayerPublic(id="Player0", money=950, folded=False, all_in=False, bet=50),
            PokerPlayerPublic(id="Player1", money=900, folded=False, all_in=False, bet=100),
        ]
        
        action = agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=150,
            current_bet=100,
            min_raise=50,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(action, PokerAction)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_agent_uses_cuda_device(self, model):
        """
        GIVEN a CUDA device
        WHEN RLAgent is created
        THEN it should work correctly on CUDA
        """
        device = torch.device("cuda")
        model = model.to(device)
        agent = RLAgent("CUDAAgent", 1000, model, device)
        
        deck = Deck()
        hole_cards = deck.draw(2)
        board = deck.draw(3)
        
        players = [
            PokerPlayerPublic(id="Player0", money=950, folded=False, all_in=False, bet=50),
            PokerPlayerPublic(id="Player1", money=900, folded=False, all_in=False, bet=100),
        ]
        
        action = agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=150,
            current_bet=100,
            min_raise=50,
            players=players,
            my_idx=0,
        )
        
        assert isinstance(action, PokerAction)


class TestRLAgentIntegration:
    """Integration tests for RLAgent in gameplay scenarios."""
    
    def test_agent_can_play_full_hand(self, rl_agent):
        """
        GIVEN an RLAgent
        WHEN simulating a full hand of poker
        THEN the agent should make valid decisions at each stage
        """
        deck = Deck()
        hole_cards = deck.draw(2)
        
        # Initial player state
        players = [
            PokerPlayerPublic(id="Hero", money=990, folded=False, all_in=False, bet=10),
            PokerPlayerPublic(id="Villain", money=980, folded=False, all_in=False, bet=20),
        ]
        
        # Pre-flop
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=[],
            pot=30,
            current_bet=20,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        assert isinstance(action, PokerAction)
        
        # Flop
        board = deck.draw(3)
        players[0] = PokerPlayerPublic(id="Hero", money=970, folded=False, all_in=False, bet=0)
        players[1] = PokerPlayerPublic(id="Villain", money=960, folded=False, all_in=False, bet=0)
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=50,
            current_bet=0,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        assert isinstance(action, PokerAction)
        
        # Turn
        board.extend(deck.draw(1))
        players[0] = PokerPlayerPublic(id="Hero", money=950, folded=False, all_in=False, bet=0)
        players[1] = PokerPlayerPublic(id="Villain", money=940, folded=False, all_in=False, bet=0)
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=70,
            current_bet=0,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        assert isinstance(action, PokerAction)
        
        # River
        board.extend(deck.draw(1))
        players[0] = PokerPlayerPublic(id="Hero", money=930, folded=False, all_in=False, bet=0)
        players[1] = PokerPlayerPublic(id="Villain", money=920, folded=False, all_in=False, bet=0)
        
        action = rl_agent.get_action(
            hole_cards=hole_cards,
            board=board,
            pot=90,
            current_bet=0,
            min_raise=20,
            players=players,
            my_idx=0,
        )
        assert isinstance(action, PokerAction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

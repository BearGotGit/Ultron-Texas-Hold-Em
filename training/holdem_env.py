# holdem_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from treys import Card

from agents.agent import PokerAgent
from training.rl_agent import RLAgent
from simulation import TexasHoldemSimulation


class TexasHoldemEnv(gym.Env):
    """
    Gymnasium environment wrapping your TexasHoldemSimulation + PokerAgent.

    - 2-player, heads-up: Hero (RLAgent) vs Villain (PokerAgent)
    - One episode = one hand
    - Each env.step() runs a full betting round for the current street
      (Pre-flop, Flop, Turn, River).

    Action space: Discrete(3)
        0 = fold
        1 = call / check
        2 = raise (pot-sized, min raise enforced)

    Observation (shape=(13,), float32):
        [hole1, hole2,
         board1, board2, board3, board4, board5,
         hero_stack, opp_stack,
         pot, to_call,
         street, dealer_btn, hero_is_sb]

        - Cards are encoded as treys int -> float.
        - street: 0=Pre, 1=Flop, 2=Turn, 3=River
        - dealer_btn: index of dealer (0 or 1)
        - hero_is_sb: 1 if hero posted small blind, else 0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, starting_chips=1000, small_blind=5, big_blind=10):
        super().__init__()

        # ----- ACTION SPACE -----
        # 0 = fold
        # 1 = call or check
        # 2 = raise (fixed-size for now)
        self.action_space = spaces.Discrete(3)

        # ----- OBSERVATION SPACE -----
        # 13 features as described above
        self.observation_space = spaces.Box(
            low=0.0,
            high=100000.0,
            shape=(14,),
            dtype=np.float32,
        )

        # ----- PLAYERS -----
        self.starting_chips = starting_chips
        self.hero = RLAgent(name="Hero", starting_chips=starting_chips)
        self.villain = PokerAgent(name="Villain", starting_chips=starting_chips)
        self.agents = [self.hero, self.villain]

        # ----- GAME ENGINE -----
        self.game = TexasHoldemSimulation(
            self.agents,
            small_blind=small_blind,
            big_blind=big_blind
        )

        # Street index: 0=Pre-flop, 1=Flop, 2=Turn, 3=River
        self.street_index = 0
        self.street_names = ["Pre-flop", "Flop", "Turn", "River"]

        # Track stack at start of the hand for reward
        self.hero_stack_start_of_hand = self.hero.get_chips()

        # Track blinds positions for hero_is_sb
        self.sb_position = None
        self.bb_position = None

        # Prevent double resolution
        self.hand_over = False
        self.pot_awarded = False

    # ----------------- CARD ENCODING & OBSERVATION -----------------

    def _encode_card(self, card):
        """
        Encode a single treys card as a float.
        If card is None (not dealt yet), return 0.0.
        """
        if card is None:
            return 0.0
        return float(card)

    def _build_obs(self):
        """
        Build the observation vector:

        [hole1, hole2,
         board1..5,
         hero_stack, opp_stack,
         pot, to_call,
         street, dealer_btn, hero_is_sb]
        """
        hero = self.hero
        villain = self.villain
        board = self.game.board  # list of treys card ints

        # ----- Hero hole cards -----
        hero_hole = hero.get_hole_cards()
        hole1 = self._encode_card(hero_hole[0]) if len(hero_hole) > 0 else 0.0
        hole2 = self._encode_card(hero_hole[1]) if len(hero_hole) > 1 else 0.0

        # ----- Board cards (pad to 5) -----
        board_cards = list(board) + [None] * (5 - len(board))
        board_enc = [self._encode_card(c) for c in board_cards]

        # ----- Stacks -----
        hero_stack = float(hero.get_chips())
        opp_stack = float(villain.get_chips())

        # ----- Pot & to_call -----
        pot = float(self.game.get_pot_size())

        # amount hero must put in to match current_bet
        to_call = float(max(0, self.game.current_bet - hero.current_bet))

        # ----- Street / dealer / hero_is_sb -----
        street = float(self.street_index)   # 0–3
        dealer_btn = float(self.game.dealer_position)

        hero_is_sb = 0.0
        if self.sb_position is not None:
            hero_is_sb = float(1.0 if self.agents[self.sb_position] is hero else 0.0)

        obs = np.array(
            [
                hole1, hole2,
                *board_enc,
                hero_stack, opp_stack,
                pot, to_call,
                street, dealer_btn, hero_is_sb
            ],
            dtype=np.float32,
        )

        return obs

    # ----------------- GYM API: RESET & STEP -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # New hand: reset engine & agents
        self.game.reset_for_new_hand()
        # dealer_position is whatever you set outside; here we can leave it as-is
        # or randomize if desired.

        # Post blinds & record positions
        self.sb_position, self.bb_position = self.game.post_blinds()

        # Deal hole cards
        self.game.deal_hole_cards()

        # Start at pre-flop
        self.street_index = 0
        self.hand_over = False
        self.pot_awarded = False
        self.hero_stack_start_of_hand = self.hero.get_chips()

        obs = self._build_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        Apply hero's action for the current street, run the entire betting
        round (including villain's decisions), then either:

        - End the hand if everyone is all-in or all but one folded, or river
          betting is done; or
        - Deal the next street and continue.

        Reward is only given at the end of the hand:
            reward = hero_stack_after_hand - hero_stack_start_of_hand
        """
        if self.hand_over:
            # If user steps after done, just return current obs + zero reward
            obs = self._build_obs()
            return obs, 0.0, True, False, {}

        # ----- Map RL action (0,1,2) → (action_str, raise_amount) -----
        action_str, raise_amount = self._map_action(action)

        # Pre-load hero's action so that when the sim calls hero.make_decision(),
        # it uses the RL-selected action.
        self.hero.set_action(action_str, raise_amount)

        # ----- Run betting round for this street -----
        street_name = self.street_names[self.street_index]
        self.game.run_betting_round(street_name)

        # ----- Check if hand is over after betting -----
        terminated = False
        truncated = False

        # If only one non-folded player remains, hand is over
        active_not_folded = [a for a in self.agents if not a.is_folded]
        if len(active_not_folded) <= 1:
            self._resolve_hand_early_fold(active_not_folded)
            terminated = True
            self.hand_over = True

        # If not over yet and we haven't reached river showdown, deal next street
        if not terminated:
            if self.street_index == 0:
                # Move to flop
                self.game.deal_flop()
                self.street_index = 1
            elif self.street_index == 1:
                # Move to turn
                self.game.deal_turn()
                self.street_index = 2
            elif self.street_index == 2:
                # Move to river
                self.game.deal_river()
                self.street_index = 3
            else:
                # We were already at river; betting round on river just finished
                # → showdown / award pot using full board
                self._resolve_hand_showdown()
                terminated = True
                self.hand_over = True

        # ----- Reward: only at the end of the hand -----
        if terminated:
            reward = float(self.hero.get_chips() - self.hero_stack_start_of_hand)
        else:
            reward = 0.0

        obs = self._build_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    # ----------------- INTERNAL HELPERS -----------------

    def _map_action(self, action_index):
        """
        Convert discrete RL action (0,1,2) into (action_str, amount).

        Remember sim.run_betting_round() interprets:
            - 'fold': ignore amount
            - 'check': ignore amount (and only legal if amount_to_call == 0)
            - 'call': ignores amount parameter; uses amount_to_call
            - 'raise': total = amount_to_call + amount
        """
        pot_size = self.game.get_pot_size()
        min_raise = self.game.min_raise

        if action_index == 0:
            return ('fold', 0)
        elif action_index == 1:
            # call / check, sim uses amount_to_call internally
            return ('call', 0)
        else:
            # raise: simple heuristic – pot-sized raise, at least min_raise
            raise_amount = max(min_raise, pot_size)
            return ('raise', raise_amount)

    def _resolve_hand_early_fold(self, active_not_folded):
        """
        Hand ended before full board (e.g., preflop all fold except one).
        In that case, we should *not* call Evaluator because board < 3.
        Just push entire pot to the one remaining player.
        """
        if self.pot_awarded:
            return

        if len(active_not_folded) == 1:
            winner = active_not_folded[0]
            amount = self.game.get_pot_size()
            winner.add_chips(amount)
            # We don't strictly need to zero pot since the hand is over,
            # but we can do it for clarity.
            self.game.pot = 0

        self.pot_awarded = True

    def _resolve_hand_showdown(self):
        """
        Hand reached showdown with at least flop dealt (board >= 3 cards).
        Use your existing award_pot() logic.
        """
        if self.pot_awarded:
            return

        # If board has < 3 cards for some reason, fall back to early-fold logic
        if len(self.game.board) < 3:
            active_not_folded = [a for a in self.agents if not a.is_folded]
            self._resolve_hand_early_fold(active_not_folded)
            return

        # Use the simulation's existing showdown logic
        # evaluate_hands() returns [(agent, score, hand_name, percentage), ...]
        _ = self.game.evaluate_hands()
        self.game.award_pot()

        self.pot_awarded = True

    # ----------------- RENDER -----------------

    def render(self):
        """
        Simple text render, reusing simulation's print_board().
        """
        print("\n=== TEXAS HOLDEM ENV RENDER ===")
        if self.game.board:
            print("Board:")
            self.game.print_board()
        else:
            print("Board: (no community cards yet)")
        print(f"Hero stack: {self.hero.get_chips()}, Villain stack: {self.villain.get_chips()}")

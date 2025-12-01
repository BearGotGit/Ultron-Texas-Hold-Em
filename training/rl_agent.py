# rl_agent.py
from agents.agent import PokerAgent

class RLAgent(PokerAgent):
    """
    PokerAgent subclass for the RL-controlled hero.
    The Gym environment sets `pending_action` before the engine
    calls make_decision().
    """

    def __init__(self, name="Hero", starting_chips=1000):
        super().__init__(name=name, starting_chips=starting_chips)
        self.pending_action = None  # (action, amount)

    def set_action(self, action, amount=0):
        """
        Called by the Gym environment to pre-load the action that
        should be taken the next time the game asks for a decision.
        """
        self.pending_action = (action, amount)

    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Override PokerAgent.make_decision.

        For the RL hero, we just return the action previously set
        by the Gym environment. If nothing is set, fall back to a
        very safe default (check/fold).
        """
        if self.is_folded or self.is_all_in:
            return ('check', 0)

        if self.pending_action is None:
            # Fallback safety: don't accidentally spew chips
            if current_bet_to_call == 0:
                return ('check', 0)
            else:
                return ('fold', 0)

        action, amount = self.pending_action
        # Clear pending action so we don't reuse it accidentally
        self.pending_action = None
        return (action, amount)
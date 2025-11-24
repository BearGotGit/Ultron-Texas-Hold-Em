"""
Texas Hold'em poker game simulation.
Manages game flow, dealing, betting rounds, and pot management.
"""

from treys import Card, Deck, Evaluator
from agents import PokerAgent


class TexasHoldemSimulation:
    """
    Simulates a complete Texas Hold'em poker game with betting.
    """
    
    def __init__(self, agents, small_blind=5, big_blind=10):
        """
        Initialize a poker game simulation.
        
        Args:
            agents: List of PokerAgent instances
            small_blind: Small blind amount (default: 5)
            big_blind: Big blind amount (default: 10)
        """
        self.agents = agents
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.evaluator = Evaluator()
        
        # Game state
        self.board = []
        self.flop = []
        self.turn = []
        self.river = []
        self.pot = 0
        self.pots = []  # List of (amount, eligible_players) for side pots
        
        # Betting state
        self.current_bet = 0  # Current bet to match in this betting round
        self.min_raise = big_blind  # Minimum raise amount
        self.dealer_position = 0
    
    def reset_for_new_hand(self):
        """Reset game state for a new hand."""
        self.deck = Deck()
        self.board = []
        self.flop = []
        self.turn = []
        self.river = []
        self.pot = 0
        self.pots = []
        self.current_bet = 0
        self.min_raise = self.big_blind
        
        for agent in self.agents:
            agent.reset_for_new_hand()
    
    def deal_hole_cards(self):
        """Deal 2 hole cards to each agent."""
        for agent in self.agents:
            hand = self.deck.draw(2)
            agent.receive_cards(hand)
    
    def post_blinds(self):
        """Post small and big blinds."""
        num_players = len(self.agents)
        sb_position = (self.dealer_position + 1) % num_players
        bb_position = (self.dealer_position + 2) % num_players
        
        # Small blind
        sb_agent = self.agents[sb_position]
        sb_amount = sb_agent.place_bet(self.small_blind)
        self.pot += sb_amount
        
        # Big blind
        bb_agent = self.agents[bb_position]
        bb_amount = bb_agent.place_bet(self.big_blind)
        self.pot += bb_amount
        self.current_bet = self.big_blind
        
        return sb_position, bb_position
    
    def deal_flop(self):
        """Deal the flop (3 community cards)."""
        self.flop = self.deck.draw(3)
        self.board = self.flop
        return self.flop
    
    def deal_turn(self):
        """Deal the turn (1 community card)."""
        self.turn = self.deck.draw(1)
        self.board = self.flop + self.turn
        return self.turn
    
    def deal_river(self):
        """Deal the river (1 community card)."""
        self.river = self.deck.draw(1)
        self.board = self.flop + self.turn + self.river
        return self.river
    
    def run_betting_round(self, round_name="Betting"):
        """
        Run a complete betting round.
        
        Args:
            round_name: Name of the betting round for display
            
        Returns:
            Boolean indicating if betting is complete (True) or hand ended (False)
        """
        print(f"\n{'='*60}")
        print(f"{round_name.upper()} ROUND")
        print(f"{'='*60}")
        print(f"Pot: ${self.pot}")
        print()
        
        # Reset current bets for new round (but keep agent.current_bet for blinds tracking)
        # Only reset self.current_bet if this is NOT pre-flop (blinds already set it)
        if round_name.lower() != "pre-flop":
            for agent in self.agents:
                agent.reset_current_bet()
            self.current_bet = 0
        
        # Determine starting position (left of dealer)
        start_position = (self.dealer_position + 1) % len(self.agents)
        
        # Track who needs to act
        active_agents = [agent for agent in self.agents if not agent.is_folded and not agent.is_all_in and agent.get_chips() > 0]
        if len(active_agents) <= 1:
            return True  # Only one player left, no betting needed
        
        # Track last aggressive action position
        last_aggressor_position = None
        position = start_position
        
        # Continue until action returns to last aggressor (or full rotation with no raises)
        while True:
            agent = self.agents[position]
            
            # Skip if folded, all-in, or no chips
            if agent.is_folded or agent.is_all_in or agent.get_chips() == 0:
                position = (position + 1) % len(self.agents)
                continue
            
            # Calculate amount to call
            amount_to_call = self.current_bet - agent.current_bet
            
            # Check if betting should end: action has returned to last aggressor
            if last_aggressor_position is not None and position == last_aggressor_position:
                break
            
            # Get agent's decision
            action, amount = agent.make_decision(
                self.board, 
                self.pot, 
                amount_to_call, 
                self.min_raise
            )
            
            # Process action
            if action == 'fold':
                agent.fold()
                print(f"{agent} folds")
                
            elif action == 'check':
                if amount_to_call == 0:
                    print(f"{agent} checks")
                    # First check sets the aggressor position (for no-bet rounds)
                    if last_aggressor_position is None:
                        last_aggressor_position = position
                else:
                    # Can't check if there's a bet - force fold
                    print(f"{agent} tried to check but must call ${amount_to_call}, folding instead")
                    agent.fold()
            
            elif action == 'call':
                actual_amount = agent.place_bet(amount_to_call)
                self.pot += actual_amount
                if agent.is_all_in:
                    print(f"{agent} calls ${actual_amount} and is ALL-IN! (chips: ${agent.get_chips()})")
                else:
                    print(f"{agent} calls ${actual_amount} (chips: ${agent.get_chips()})")
                
                # First action (if no bet) sets aggressor
                if last_aggressor_position is None and self.current_bet > 0:
                    last_aggressor_position = position
            
            elif action == 'raise':
                # Total amount is call + raise
                total_raise = amount_to_call + amount
                actual_amount = agent.place_bet(total_raise)
                self.pot += actual_amount
                
                # Update current bet
                old_bet = self.current_bet
                self.current_bet = agent.current_bet
                
                # Only update aggressor if they actually raised
                if self.current_bet > old_bet:
                    last_aggressor_position = position
                    self.min_raise = max(amount, self.min_raise)
                
                if agent.is_all_in:
                    print(f"{agent} raises to ${agent.current_bet} and is ALL-IN!")
                else:
                    print(f"{agent} raises to ${agent.current_bet} (chips: ${agent.get_chips()})")
            
            # Check if hand is over (only 1 or 0 active players)
            active_players_remaining = [a for a in self.agents if not a.is_folded and (not a.is_all_in or a.get_chips() > 0)]
            non_allin_remaining = [a for a in active_players_remaining if not a.is_all_in and a.get_chips() > 0]
            
            if len(active_players_remaining) <= 1:
                print(f"\nPot after {round_name}: ${self.pot}\n")
                return True
            
            # If only one non-all-in player remains, they must match the current bet or fold
            # Don't exit early - let them act on any outstanding bets
            if len(non_allin_remaining) == 1:
                # Check if this player has matched the current bet
                remaining_player = non_allin_remaining[0]
                if remaining_player.current_bet >= self.current_bet:
                    # They've matched, betting is complete
                    print(f"\nPot after {round_name}: ${self.pot}\n")
                    return True
                # Otherwise, continue the loop so they can act
            elif len(non_allin_remaining) == 0:
                # Everyone is all-in or folded, no more betting possible
                print(f"\nPot after {round_name}: ${self.pot}\n")
                return True
            
            # Move to next position
            position = (position + 1) % len(self.agents)
        
        print(f"\nPot after {round_name}: ${self.pot}\n")
        return True
    
    def calculate_all_equities(self):
        """
        Calculate equity for all agents.
        
        Returns:
            List of equity values for each agent
        """
        all_hands = [agent.get_hole_cards() for agent in self.agents]
        remaining_cards = self.deck.cards[:]
        
        # Use first agent's equity calculator (they all use same algorithm)
        equities = self.agents[0]._calculate_all_equities(
            self.board, 
            all_hands, 
            remaining_cards, 
            num_simulations=1000
        )
        return equities
    
    def evaluate_hands(self):
        """
        Evaluate all hands and determine winner.
        
        Returns:
            List of tuples: (agent, score, hand_name, percentage)
        """
        results = []
        for agent in self.agents:
            score, hand_class, hand_name, percentage = agent.evaluate_hand(self.board)
            results.append((agent, score, hand_name, percentage))
        
        return results
    
    def get_winner(self):
        """
        Determine the winner(s) of the game.
        
        Returns:
            List of winning agents
        """
        results = self.evaluate_hands()
        best_score = min(result[1] for result in results if not result[0].is_folded)
        winners = [result[0] for result in results if result[1] == best_score and not result[0].is_folded]
        return winners
    
    def create_side_pots(self):
        """
        Create side pots based on all-in situations.
        
        Returns:
            List of tuples: (pot_amount, eligible_agents)
        """
        # Get all non-folded players with their total investments
        active_players = [(agent, agent.total_invested) for agent in self.agents if not agent.is_folded]
        
        if len(active_players) <= 1:
            return [(self.pot, [agent for agent, _ in active_players])]
        
        # Sort by investment amount
        active_players.sort(key=lambda x: x[1])
        
        # Check if all players invested the same amount (no all-ins)
        if all(inv == active_players[0][1] for _, inv in active_players):
            # Single pot, everyone is eligible
            return [(self.pot, [agent for agent, _ in active_players])]
        
        pots = []
        remaining_pot = self.pot
        previous_investment = 0
        eligible_players = [agent for agent, _ in active_players]
        
        for i, (agent, investment) in enumerate(active_players):
            if investment > previous_investment and remaining_pot > 0:
                # Calculate pot for this level
                contribution_per_player = investment - previous_investment
                pot_amount = contribution_per_player * len(eligible_players)
                pot_amount = min(pot_amount, remaining_pot)
                
                pots.append((pot_amount, eligible_players[:]))
                remaining_pot -= pot_amount
                previous_investment = investment
            
            # Remove this player from future pots (they're all-in at this level)
            if i < len(active_players) - 1:
                eligible_players.remove(agent)
        
        return pots
    
    def award_pot(self):
        """
        Award pot(s) to winner(s) with proper side pot handling.
        
        Returns:
            List of tuples: (winner, amount_won, pot_description)
        """
        # Create side pots
        side_pots = self.create_side_pots()
        
        results = []
        
        # Award each pot separately
        for pot_num, (pot_amount, eligible_players) in enumerate(side_pots):
            if pot_amount == 0 or len(eligible_players) == 0:
                continue
            
            # Evaluate hands for eligible players only
            pot_results = []
            for agent in eligible_players:
                score, hand_class, hand_name, percentage = agent.evaluate_hand(self.board)
                pot_results.append((agent, score, hand_name))
            
            # Find winner(s) for this pot
            best_score = min(result[1] for result in pot_results)
            pot_winners = [result[0] for result in pot_results if result[1] == best_score]
            
            # Split pot among winners
            winnings_per_winner = int(pot_amount / len(pot_winners))
            
            # First pot is main pot, subsequent pots are side pots
            pot_description = "Main pot" if pot_num == 0 else f"Side pot {pot_num}"
            
            for winner in pot_winners:
                winner.add_chips(winnings_per_winner)
                results.append((winner, winnings_per_winner, pot_description))
        
        return results
    
    def get_pot_size(self):
        """Get current pot size."""
        return self.pot
    
    def print_board(self):
        """Print the current board cards."""
        if self.board:
            Card.print_pretty_cards(self.board)
        else:
            print("No community cards dealt yet")
    
    def print_game_state(self):
        """Print complete game state including all hands."""
        print(f"\n{'='*60}")
        print("GAME STATE")
        print(f"{'='*60}\n")
        
        print("Community Cards:")
        self.print_board()
        print()
        
        for i, agent in enumerate(self.agents):
            print(f"{agent} hole cards:")
            Card.print_pretty_cards(agent.get_hole_cards())
            print()

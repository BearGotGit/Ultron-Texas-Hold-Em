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
        
        # Reset current bets for new round
        for agent in self.agents:
            agent.reset_current_bet()
        self.current_bet = 0
        
        # Determine starting position (left of dealer)
        start_position = (self.dealer_position + 1) % len(self.agents)
        
        # Track who needs to act
        active_agents = [agent for agent in self.agents if not agent.is_folded and not agent.is_all_in]
        if len(active_agents) <= 1:
            return True  # Only one player left, no betting needed
        
        # Continue until all active players have matched the current bet
        actions_this_round = 0
        last_raiser_position = None
        position = start_position
        
        while True:
            agent = self.agents[position]
            
            # Skip if folded or all-in
            if agent.is_folded or agent.is_all_in:
                position = (position + 1) % len(self.agents)
                continue
            
            # Check if betting is complete
            if last_raiser_position is not None and position == last_raiser_position:
                break
            if actions_this_round >= len(active_agents) and all(
                a.current_bet == self.current_bet or a.is_all_in or a.is_folded 
                for a in self.agents
            ):
                break
            
            # Calculate amount to call
            amount_to_call = self.current_bet - agent.current_bet
            
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
                active_agents = [a for a in active_agents if not a.is_folded]
                if len(active_agents) <= 1:
                    return True
            
            elif action == 'check':
                if amount_to_call == 0:
                    print(f"{agent} checks")
                else:
                    print(f"{agent} tried to check but must call ${amount_to_call}, folding instead")
                    agent.fold()
                    active_agents = [a for a in active_agents if not a.is_folded]
                    if len(active_agents) <= 1:
                        return True
            
            elif action == 'call':
                actual_amount = agent.place_bet(amount_to_call)
                self.pot += actual_amount
                print(f"{agent} calls ${actual_amount} (chips: ${agent.get_chips()})")
            
            elif action == 'raise':
                # Total amount is call + raise
                total_raise = amount_to_call + amount
                actual_amount = agent.place_bet(total_raise)
                self.pot += actual_amount
                
                # Update current bet
                self.current_bet = agent.current_bet
                self.min_raise = amount
                last_raiser_position = position
                
                print(f"{agent} raises to ${agent.current_bet} (chips: ${agent.get_chips()})")
            
            actions_this_round += 1
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
    
    def award_pot(self):
        """
        Award the pot to the winner(s).
        
        Returns:
            List of winners and their winnings
        """
        winners = self.get_winner()
        winnings_per_player = self.pot / len(winners)
        
        results = []
        for winner in winners:
            winner.add_chips(int(winnings_per_player))
            results.append((winner, int(winnings_per_player)))
        
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

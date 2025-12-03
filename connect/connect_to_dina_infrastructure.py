"""
Connect to Dina's Texas Hold'em WebSocket server.

Server info:
- URL: ws://localhost:8080/ws?apiKey=dev&table=table-1&player=YourPlayerName
- Protocol: JSON over WebSocket
- Server is host-authoritative (validates all actions)

Protocol:
    Client → Server:
        Join:   {"type": "join"}
        Action: {"type": "action", "action": "call"}
        Raise:  {"type": "action", "action": "raise", "amount": 120}
    
    Server → Client:
        State:  {"type": "state", "hand": 2, "phase": "TURN", "pot": 540, ...}
        Error:  {"type": "error", "message": "Not your turn"}
    
    Phases: WAITING, PREFLOP, FLOP, TURN, RIVER, SHOWDOWN

Usage:
    python connect_to_dina_infrastructure.py --player YourName --checkpoint path/to/model.pt
    python connect_to_dina_infrastructure.py --player YourName --random
"""

import asyncio
import json
from agents.rl_agent import RLAgent
import websockets
import argparse

# ===== CLIENT =====

# Init agent
# Option 1: Random agent
from agents.monte_carlo_agent import RandomAgent
agent = RandomAgent(player_id="UltronRandom", starting_money=1000)

# Option 2: RL agent from checkpoint
# from agents.rl_agent import RLAgent
# from training.train_rl_model import PPOConfig  # Needed for unpickling
# agent = RLAgent.from_checkpoint(
#     checkpoint_path="checkpoints/2025-12-1/100k/final.pt",
#     player_id="YourBotName",
#     starting_money=1000,
# )

# ===== CONNECT =====

async def play_poker():
    API_KEY = "dev"
    TABLE = "table-1"
    PLAYER = agent.player_id  # Use agent's player_id
    
    url = f"ws://localhost:8080/ws?apiKey={API_KEY}&table={TABLE}&player={PLAYER}"
    
    async with websockets.connect(url) as ws:
        # Join the table
        await ws.send(json.dumps({"type": "join"}))
        print(f"✓ Connected as {PLAYER}")
        
        # Main game loop
        while True:
            # Receive message from server
            msg_str = await ws.recv()
            msg = json.loads(msg_str)
            
            print(f"Received: {msg['type']}")
            
            # Handle different message types
            if msg['type'] == 'error':
                print(f"❌ Error: {msg['message']}")
                continue
            
            if msg['type'] == 'state':
                # Server sent game state update
                # Available fields: hand, phase, pot, current_player, hole_cards, community_cards, players, etc.
                phase = msg.get('phase')
                pot = msg.get('pot', 0)
                
                print(f"Hand {msg.get('hand')}, Phase: {phase}, Pot: ${pot}")
                
                # Wait for PREFLOP, FLOP, TURN, or RIVER to act (skip WAITING/SHOWDOWN)
                if phase not in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
                    continue
                
                # TODO: Check if it's our turn
                # if msg.get('current_player') != PLAYER:
                #     continue
                
                # TODO: Convert server observation to our format
                # hole_cards = convert_cards(msg.get('hole_cards', []))  # Convert to Treys integers
                # board = convert_cards(msg.get('community_cards', []))
                # current_bet = msg.get('current_bet', 0)
                # min_raise = msg.get('min_raise', 0)
                # players = convert_players(msg.get('players', []))  # Convert to PokerPlayerPublic list
                # my_idx = find_my_index(players, PLAYER)
                # 
                # # Get action from agent
                # action = agent.get_action(
                #     hole_cards=hole_cards,
                #     board=board,
                #     pot=pot,
                #     current_bet=current_bet,
                #     min_raise=min_raise,
                #     players=players,
                #     my_idx=my_idx
                # )
                # # Returns: PokerAction(action_type=ActionType.FOLD/CHECK/CALL/RAISE, amount=...)
                # 
                # # Convert to server format and send
                # # Server expects: {"type": "action", "action": "call"}
                # #            or:  {"type": "action", "action": "raise", "amount": 120}
                # if action.action_type.value == 'raise':
                #     server_action = {
                #         'type': 'action',
                #         'action': 'raise',
                #         'amount': action.amount
                #     }
                # else:
                #     # fold, check, or call
                #     server_action = {
                #         'type': 'action',
                #         'action': action.action_type.value
                #     }
                # 
                # await ws.send(json.dumps(server_action))
                # print(f"➤ Sent: {server_action['action']}{' $' + str(server_action.get('amount', '')) if 'amount' in server_action else ''}")

# ===== HELPER FUNCTIONS =====

def convert_cards(card_strings):
    """
    Convert server card format to Treys integers.
    
    Server format: ["Ah", "Kd", "Qs", ...]
    Treys format: integers (use Card.new() from treys)
    
    Example:
        from treys import Card
        return [Card.new(card_str) for card_str in card_strings]
    """
    pass

def convert_players(server_players):
    """
    Convert server player list to PokerPlayerPublic list.
    
    Server format: [{"name": "p1", "chips": 1000, "bet": 50, "folded": false, ...}, ...]
    Our format: [PokerPlayerPublic(player_id, money, folded, all_in, bet), ...]
    
    Example:
        from agents.poker_player import PokerPlayerPublic
        return [
            PokerPlayerPublic(
                player_id=p['name'],
                money=p['chips'],
                folded=p['folded'],
                all_in=p.get('all_in', False),
                bet=p['bet']
            )
            for p in server_players
        ]
    """
    pass

def find_my_index(players, my_name):
    """Find our index in the player list."""
    for i, player in enumerate(players):
        if player.player_id == my_name:
            return i
    return 0

# ===== MAIN =====

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Connect bot to Dina's poker server")
    parser.add_argument('--player', required=True, help='Your player name')
    parser.add_argument('--checkpoint', help='Path to RL model checkpoint')
    parser.add_argument('--random', action='store_true', help='Use random agent instead')
    parser.add_argument('--api-key', default='dev', help='API key for server')
    parser.add_argument('--table', default='table-1', help='Table name')
    args = parser.parse_args()
    
    # Initialize agent based on args
    if args.random:
        agent = RandomAgent(player_id=args.player, starting_money=1000)
    else:
        agent = RLAgent.from_checkpoint(
            checkpoint_path=args.checkpoint,
            player_id=args.player,
            starting_money=1000,
        )
    
    # Run the async game loop
    asyncio.run(play_poker())

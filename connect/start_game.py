import asyncio
import websockets
import json
import os
from dotenv import load_dotenv

async def start_hand():
    # Load environment variables
    load_dotenv()
    
    base_url = os.getenv("BASEURL", "ws://localhost:8080")
    api_key = os.getenv("APIKEY", "dev")
    start_key = os.getenv("STARTKEY", "supersecret")
    table_id = os.getenv("TABLEID", "table-2")
    
    # Build WebSocket URL with both keys
    # The host player name can be anything
    ws_url = f"{base_url}/ws?apiKey={api_key}&startKey={start_key}&table={table_id}&player=host"
    
    print(f"Connecting to: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("Connected as host!")
            
            # Send the host_start message
            start_message = {
                "type": "host_start"
            }
            
            await websocket.send(json.dumps(start_message))
            print("Sent host_start message")
            
            # Wait a bit to receive confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                print("No immediate response (this is normal)")
            
            print("Hand should be started!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(start_hand())
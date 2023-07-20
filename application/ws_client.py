import asyncio
import time
from application.config import ORCHESTRATOR_WS_ADDRESS, WS_TIMEOUT
from datamodels.models import WebsocketResponse
import websockets
from logging import log, ERROR, INFO, DEBUG
import os

async def websocket_client():
    address = ORCHESTRATOR_WS_ADDRESS
    #Connect to the websocket server on FL Orchestrator
    try:
        async for websocket in websockets.connect(address, open_timeout=2, close_timeout=2):
            while True:
                try:
                    # Check what is the value of the current status
                    log(INFO, f'Websocket port available on {websocket.local_address}')
                    status = os.getenv('FL_LO_STATE')
                    # Turn it into a JSON and send
                    await asyncio.sleep(WS_TIMEOUT)
                    await websocket.send(status)

                    # Wait the server response
                    #response = await websocket.recv()
                    # Add logs here
                except ConnectionRefusedError as e:
                    log(ERROR, f'Websocket connection refused - cannot connect to server websocket')
                except websockets.ConnectionClosed:
                    break
    except ConnectionRefusedError as e:
        log(ERROR, f'Websocket connection refused - cannot connect to server websocket')
    except websockets.exceptions.ConnectionClosedError as e:
        log(ERROR, f'Websocket connection closed improperly')
    except Exception as e:
        log(ERROR, e)
    websocket_client()

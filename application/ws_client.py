import asyncio
import time

from prometheus_client import Info, Counter
from application.config import ORCHESTRATOR_WS_ADDRESS, WS_TIMEOUT
from datamodels.models import WebsocketResponse
import websockets
from logging import log, ERROR, INFO, DEBUG
import os
i = Info('fl_state', 'Current state of FL Local Operations')
c = Counter('websocket_conn_count', 'How many times the websocket client connection was started/restarted')

async def websocket_client():
    address = ORCHESTRATOR_WS_ADDRESS
    #Connect to the websocket server on FL Orchestrator
    try:
        async for websocket in websockets.connect(address):
            log(INFO, f'Websocket port available on {websocket.local_address}')
            c.inc()
            while True:
                try:
                    # Check what is the value of the current status
                    status = os.getenv('FL_LO_STATE')
                    i.info({'state': status})
                    # Turn it into a JSON and send
                    await asyncio.sleep(WS_TIMEOUT)
                    await websocket.send(status)

                    # Wait the server response
                    #response = await websocket.recv()
                    # Add logs here
                except ConnectionRefusedError as e:
                    log(ERROR, f'Websocket connection refused - cannot connect to server websocket')
                except websockets.ConnectionClosed:
                    continue
    except ConnectionRefusedError as e:
        log(ERROR, f'Websocket connection refused - cannot connect to server websocket')
    except websockets.exceptions.ConnectionClosedError as e:
        log(ERROR, f'Websocket connection closed improperly')
    except Exception as e:
        log(ERROR, e)
    websocket_client()

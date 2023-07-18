import time
from application.config import ORCHESTRATOR_WS_ADDRESS, WS_TIMEOUT
from datamodels.models import WebsocketResponse
import websockets
from logging import log, INFO, DEBUG
import os

async def websocket_client():
    #Connect to the websocket server on FL Orchestrator
    address = ORCHESTRATOR_WS_ADDRESS
    async with websockets.connect(address) as websocket: #Define the address and port
        # Check what is the value of the current status
        log(INFO, f'Websocket port available on {websocket.local_address}')
        status = os.getenv('FL_LO_STATE')
        # Turn it into a JSON and send
        time.sleep(WS_TIMEOUT)
        await websocket.send(status)

        # Wait the server response
        #response = await websocket.recv()
        # Add logs here

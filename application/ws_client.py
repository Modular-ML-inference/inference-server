import asyncio
import json
import socket

from prometheus_client import Info, Counter
from application.config import ORCHESTRATOR_WS_ADDRESS, WS_TIMEOUT
import websockets
from logging import log, ERROR, INFO
import os
i = Info('fl_state', 'Current state of FL Local Operations')
c = Counter('websocket_conn_count',
            'How many times the websocket client connection was started/restarted')


def check_ip_and_port():
    """A small helper function that checks whether the port and IP values 
    have been set and returns the pod ports and ips otherwise"""
    IP_KEYWORD = "FL_LO_IP"
    PORT_KEYWORD = "FL_LO_PORT"
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    port = '9050'
    if IP_KEYWORD in os.environ:
        ip_addr = os.environ[IP_KEYWORD]
    if PORT_KEYWORD in os.environ:
        port = os.environ[PORT_KEYWORD]
    return ip_addr, port


async def websocket_client():
    address = ORCHESTRATOR_WS_ADDRESS
    # Connect to the websocket server on FL Orchestrator
    try:
        async for websocket in websockets.connect(address, ping_interval=None):
            log(INFO, f'Websocket port available on {websocket.local_address}')
            c.inc()
            while True:
                try:
                    # Check what is the value of the current status
                    status = os.getenv('FL_LO_STATE')
                    ip, port = check_ip_and_port()
                    message = {"host": ip, "port": port, "status": status}
                    prepared = json.dumps(message)
                    i.info({'state': status})
                    # Turn it into a JSON and send
                    await asyncio.sleep(WS_TIMEOUT)
                    await websocket.send(prepared)

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

# datahub is a hub transmitting data between Unreal Project
# and other component via websocket.
# ref: https://websockets.readthedocs.io/en/stable/howto/patterns.html
import asyncio
import websockets
import json
import re
import settings
from service import Service

# async def handle(websocket):
#     async for message in websocket:
#         print(message)
#         await websocket.send(message)

# Here we use simple consumer and producer pattern to send/recv message on same socket
# consumer: the function to handle received message
# producer: the function to handle sent message

class DataHubService(Service):
    debug = False 

    def __init__(self, host, port, datalog=None):
        super(DataHubService).__init__()
        self.host = host
        self.port = port
        self.datalog = datalog

    # on message recv
    async def consumer(self, msg):
        msg = self.parse_message(msg)
        if self.datalog:
            for data in msg:
                self.datalog.enqueue(data)

    # on message sent
    async def producer(self):
        await asyncio.sleep(2)
        msg = "heart beat"
        if self.debug:
            self.print(f"[DBG] >>> {msg}")
        return msg

    async def consumer_handler(self, websocket):
        async for msg in websocket:
            await self.consumer(msg)

    async def producer_handler(self, websocket):
        while True:
            msg = await self.producer()
            await websocket.send(msg)

    async def handler(self, websocket):
        consumer_task = asyncio.create_task(self.consumer_handler(websocket))
        producer_task = asyncio.create_task(self.producer_handler(websocket))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    async def core(self):
        self.print("[INF] starting...")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def parse_message(self, msg):
        results = []
        try:
            msg_list = re.split('({[^}]*})', msg)[1::2]
            for m in msg_list:
                data = json.loads(m)
                data = {k.lower(): v for k, v in data.items()} # convert to lower
                if self.debug:
                    self.print("[DBG] <<< " + str(data))
                results.append(data)
        except:
            self.print(f'[ERR] fail to parse:\n {msg}\n')
        return results

if __name__ == "__main__":
    HOST, PORT = settings.DATAHUB_CONFIG['endpoint']

    datahub = DataHubService(host=HOST, port=PORT)
    datahub.run()

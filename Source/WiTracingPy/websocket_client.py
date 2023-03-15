import asyncio
import websockets

async def hello():
    async with websockets.connect("ws://localhost:8080") as websocket:
        await websocket.send("Hello world!")
        data = await websocket.recv()
        print(data)

asyncio.run(hello())
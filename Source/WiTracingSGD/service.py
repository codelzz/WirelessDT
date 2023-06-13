import asyncio

class Service:
    def __init__(self):
        pass

    async def do(self):
        await asyncio.sleep(1)
        self.print("[INF] working...")

    async def core(self):
        self.print("[INF] starting...")
        while True:
            await self.do()
            
    def task(self):
        return asyncio.create_task(self.core())

    # run as standalone
    def run(self):
        asyncio.run(self.core())

    def print(self, payload):
        print(f"{f'[{self.__class__.__name__}]':18} {payload}")

if __name__ == "__main__":
    service = Service()

    # Approach #1: run as standalone
    # service.run()

    # Approach #2: run as task
    async def services():
        done, pending = await asyncio.wait(
            [service.task()],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    asyncio.run(services())
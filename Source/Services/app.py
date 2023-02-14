import asyncio
import settings
from datahub import DataHubService
from datalog import DataLogService

if __name__ == "__main__":
    HOST, PORT = settings.DATAHUB_CONFIG['endpoint']
    # data log
    datalog = DataLogService()
    # data hub
    datahub = DataHubService(host=HOST, port=PORT, datalog=datalog)
    datahub.debug = True

    async def services():
        done, pending = await asyncio.wait(
            [datahub.task(), datalog.task()],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    asyncio.run(services())